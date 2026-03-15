import json
import boto3
from mcp.server.fastmcp import FastMCP

# Create the FastMCP server
mcp = FastMCP(
    name="chronos-forecasting",
    host="0.0.0.0",
    port=8002
)

# Register the tool with the FastMCP server
@mcp.tool()
def generate_forecasts(
    target: list[float],
    prediction_length: int,
    quantile_levels: list[float]
) -> dict:
    """
    Generate probabilistic time series forecasts using Chronos on Amazon Bedrock.

    Parameters:
    ===============================================================================
    target: list of float.
        The historical time series values used as context.
    
    prediction_length: int.
        The number of future time steps to predict.

    quantile_levels: list of float.
        The quantiles to be predicted at each future time step.

    Returns:
    ===============================================================================
        Dictionary with predicted mean and quantiles at each future time step.
    """
    # Create the Bedrock client
    bedrock_runtime_client = boto3.client("bedrock-runtime")
    
    # Invoke the Bedrock endpoint
    response = bedrock_runtime_client.invoke_model(
        modelId="<bedrock-endpoint-arn>",
        body=json.dumps({
            "inputs": [{
                "target": target
            }],
            "parameters": {
                "prediction_length": prediction_length,
                "quantile_levels": quantile_levels
            }
        })
    )
    
    # Extract the forecasts
    forecasts = json.loads(response["body"].read()).get("predictions")[0]
    
    # Return the forecasts
    return forecasts


# Run the FastMCP server with SSE transport
if __name__ == "__main__":
    mcp.run(transport="sse")
