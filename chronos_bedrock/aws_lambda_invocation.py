import io
import json
import boto3
import pandas as pd

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
