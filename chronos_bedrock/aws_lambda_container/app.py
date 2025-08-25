import json
import boto3
import pandas as pd
import clickhouse_connect


def handler(event, context):
    """
    Generate zero-shot forecasts with Chronos-Bolt (Base) Amazon Bedrock endpoint using data stored in ClickHouse.

    Parameters:
    ========================================================================================================
    event: dict.
        A dictionary with the following keys:

        initialization_timestamp: str.
            The initialization timestamp of the forecasts, in ISO format (YYYY-MM-DD HH:mm:ss).

        frequency: int.
            The frequency of the time series, in minutes.

        context_length: int.
            The number of past time steps to use as context.

        prediction_length: int.
            The number of future time steps to predict.

        quantile_levels: list of float.
            The quantiles to be predicted at each future time step.

    context: AWS Lambda context object, see https://docs.aws.amazon.com/lambda/latest/dg/python-context.html.
    """
    # Create the ClickHouse client
    clickhouse_client = clickhouse_connect.get_client(
        host="<clickhouse-host>",
        user="<clickhouse-user>",
        password="<clickhouse-password>",
        secure=True
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
    
    # Create the Bedrock client
    bedrock_runtime_client = boto3.client(
        service_name="bedrock-runtime"
    )
    
    # Invoke the Bedrock endpoint with the ClickHouse data
    response = bedrock_runtime_client.invoke_model(
        modelId="<bedrock-endpoint-arn>",
        body=json.dumps({
            "inputs": [{
                "target": df["total_load"].values.tolist(),
            }],
            "parameters": {
                "prediction_length": event["prediction_length"],
                "quantile_levels": event["quantile_levels"],
            }
        })
    )
    
    # Extract the forecasts
    predictions = json.loads(response["body"].read()).get("predictions")[0]
    
    # Add the timestamps to the forecasts
    predictions = {
                      "timestamp": [
                          x.strftime("%Y-%m-%d %H:%M:%S")
                          for x in pd.date_range(
                              start=event["initialization_timestamp"],
                              periods=event["prediction_length"],
                              freq=f"{event['frequency']}min",
                          )
                      ]
                  } | predictions
    
    # Return the forecasts
    return {
        "statusCode": 200,
        "body": json.dumps(predictions)
    }