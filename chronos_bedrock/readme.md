# Zero-Shot Time Series Forecasting with Chronos-Bolt using Amazon Bedrock and ClickHouse


![png](https://clickhouse-aws-ml-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/architecture_diagram.png)



Chronos is a family of foundation models for probabilistic time series forecasting ...

![png](https://clickhouse-aws-ml-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/chronos_architecture.png)





## 1. Create the Bedrock model endpoint
```python
import boto3

bedrock_client = boto3.client("bedrock")

response = bedrock_client.create_marketplace_model_endpoint(
    modelSourceIdentifier=f"arn:aws:sagemaker:'<bedrock-region>':aws:hub-content/SageMakerPublicHub/Model/autogluon-forecasting-chronos-bolt-base/2.0.6",
    endpointConfig={
        "sageMaker": {
            "initialInstanceCount": 1,
            "instanceType": "ml.m5.4xlarge",
            "executionRole": '<bedrock-execution-role>'
        }
    },
    acceptEula=True,
    endpointName="chronos-bolt-base-endpoint",
)

bedrock_endpoint_arn = response["marketplaceModelEndpoint"]["endpointArn"]
```

## 2. Create the Lambda function for invoking the model endpoint with ClickHouse data

**`app.py`**

```python
import json
import boto3
import pandas as pd
import clickhouse_connect

def handler(event, context):
    """
    Generate zero-shot forecasts with Chronos-Bolt Amazon Bedrock endpoint using data stored in ClickHouse.

    Parameters:
    ========================================================================================================
    event: dict.
        A dictionary with the following keys:
        
        initialization_timestamp: str.
            The first timestamp for which the forecasts are generated, in ISO format (YYYY-MM-DD HH:mm:ss).
            
        frequency: int.
            The frequency of the time series, in minutes.
        
        context_length: int.
            The number of past time steps to use as context.
        
        prediction_length: int.
            The number of future time steps to forecast.

    context: AWS Lambda context object, see https://docs.aws.amazon.com/lambda/latest/dg/python-context.html.
    
    Returns:
    ========================================================================================================
    A JSON body with the following keys:
        
        timestamp: str.
            The future timestamp.
            
        mean: float:
            The predicted mean.
        
        0.1: float.
            The predicted 10th percentile.
        
        0.5: float.
            The predicted median.
            
        0.9: float.
            The predicted 90th percentile.
    """
    # Create the ClickHouse client
    clickhouse_client = clickhouse_connect.get_client(
        host="<clickhouse-host>",
        user="<clickhouse-user>",
        password="<clickhouse-password>",
        secure=True
    )
    
    # Create the Bedrock client
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime"
    )
    
    # Load the input data from ClickHouse
    df = clickhouse_client.query_df(
        f"""
            select
                timestamp,
                total_load
            from
                total_load_data
            where
                timestamp < toDateTime('{event['initialization_timestamp']}')
            and
                timestamp >= toDateTime('{event['initialization_timestamp']}') - INTERVAL {int(event['frequency']) * int(event['context_length'])} MINUTES
            order by
                timestamp asc
        """
    )
    print(f"""
        Loaded {len(df)} records from ClickHouse.
        Start timestamp: {min(df['timestamp'])}.
        End timestamp: {max(df['timestamp'])}.
    """)
    
    # Invoke the Bedrock endpoint with the ClickHouse data
    response = bedrock_runtime_client.invoke_model(
        modelId="<bedrock-endpoint-arn>",
        body=json.dumps({
            "inputs": [{
                "target": df["total_load"].values.tolist(),
            }],
            "parameters": {
                "prediction_length": event["prediction_length"],
                "quantile_levels": [0.1, 0.5, 0.9],
            }
        })
    )
    
    # Extract the forecasts
    predictions = json.loads(response["body"].read()).get("predictions")[0]
    
    # Add the timestamps to the forecasts
    predictions["timestamp"] = [
        x.strftime("%Y-%m-%d %H:%M:%S")
        for x in pd.date_range(
            start=event["initialization_timestamp"],
            periods=event["prediction_length"],
            freq=f"{event['frequency']}min",
        )
    ]
    
    print(f"""
        Received {len(predictions['mean'])} forecasts from Bedrock.
        Start timestamp: {min(predictions['timestamp'])}.
        End timestamp: {max(predictions['timestamp'])}.
    """)
    
    # Return the forecasts
    return {
        "statusCode": 200,
        "body": json.dumps(predictions)
    }
```

**`requirements.txt`**

```
boto3==1.34.84
clickhouse_connect==0.8.18
pandas==2.3.1
```

**`Dockerfile`**

```
FROM amazon/aws-lambda-python:3.12

COPY requirements.txt  .

RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY app.py ${LAMBDA_TASK_ROOT}

CMD ["app.handler"]
```

**`build_and_push.sh`**

```commandline
aws_account_id='<aws-account-id>'
region='<ecr-repository-region>'
algorithm_name='<ecr-repository-name>'

aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $aws_account_id.dkr.ecr.$region.amazonaws.com

docker build -t $algorithm_name .

docker tag $algorithm_name:latest $aws_account_id.dkr.ecr.$region.amazonaws.com/$algorithm_name:latest

docker push $aws_account_id.dkr.ecr.$region.amazonaws.com/$algorithm_name:latest
```

## 3. Test the Lambda function and compare the forecasts to the historical data stored in ClickHouse

```python
import io
import json
import boto3
import pandas as pd
import clickhouse_connect
```

```python
def invoke_lambda_function(
    initialization_timestamp,
    frequency,
    context_length,
    prediction_length,
    function_name
):
    """
    Invoke the Lambda function that generates zero-shot forecasts with Chronos-Bolt
    Amazon Bedrock endpoint using the data stored in ClickHouse.
    
    Parameters:
    ========================================================================================================
    initialization_timestamp: str.
        The first timestamp for which the forecasts are generated, in ISO format (YYYY-MM-DD HH:mm:ss).
        
    frequency: int.
        The frequency of the time series, in minutes.
    
    context_length: int.
        The number of past time steps to use as context.
    
    prediction_length: int.
        The number of future time steps to forecast.
    
    function_name: str.
        The name of the Lambda function.
    
    Returns:
    ========================================================================================================
    forecasts: pd.DataFrame.
    
        A data frame with the following columns:
        
        timestamp: str.
            The future timestamp.
            
        mean: float:
            The predicted mean.
        
        0.1: float.
            The predicted 10th percentile.
        
        0.5: float.
            The predicted median.
            
        0.9: float.
            The predicted 90th percentile.
    """
    # Create the Lambda client
    lambda_client = boto3.client("lambda")
    
    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName=function_name,
        Payload=json.dumps({
            "initialization_timestamp": initialization_timestamp,
            "frequency": frequency,
            "prediction_length": prediction_length,
            "context_length": context_length
        })
    )
    
    # Extract the forecasts in a data frame
    forecasts = pd.read_json(io.StringIO(json.loads(response["Payload"].read())["body"]))
    
    # Return the forecasts
    return forecasts
```

```python
frequency=15
context_length=24 * 4 * 7 * 3
prediction_length=24 * 4
function_name = "chronos-bedrock"
```

```python
predictions = invoke_lambda_function(
    initialization_timestamp="2025-08-17 00:00:00",
    frequency=frequency,
    context_length=context_length,
    prediction_length=prediction_length,
    function_name=function_name
)
```

```python
forecasts = invoke_lambda_function(
    initialization_timestamp="2025-08-18 00:00:00",
    frequency=frequency,
    context_length=context_length,
    prediction_length=prediction_length,
    function_name=function_name
)
```



```python
clickhouse_client = clickhouse_connect.get_client(
    host="<clickhouse-host>",
    user="<clickhouse-user>",
    password="<clickhouse-password>",
    secure=True
)

df = clickhouse_client.query_df(
    """
    select
        timestamp,
        total_load
    from
        total_load_data
    where
        timestamp >= toDateTime('2025-08-18 23:45:00') - INTERVAL 14 DAYS
    order by
        timestamp asc
    """
)

output = pd.merge(
    left=df,
    right=pd.concat([predictions, forecasts], axis=0),
    on="timestamp",
    how="outer"
)
```

![png](https://clickhouse-aws-ml-blog.s3.eu-west-2.amazonaws.com/chronos_bedrock/chronos_bedrock_zero_shot_forecasts.png)