import json
import boto3
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ── Tools ──────────────────────────────────────────────────────────────────

# Create the Bedrock runtime client
bedrock_runtime_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="eu-west-1"
)


# Define the time series forecasting tool
@tool
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
    dict
        Dictionary with predicted mean and quantiles at each future time step.
    """
    # Invoke the Chronos endpoint on Amazon Bedrock
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
    
    # Extract and return the forecasts
    forecasts = json.loads(response["body"].read()).get("predictions")[0]
    return forecasts


# ── Agent ──────────────────────────────────────────────────────────────────

# Create the agent
agent = Agent(
    model="eu.anthropic.claude-sonnet-4-6",
    tools=[generate_forecasts],
    system_prompt=(
        "You are a time series forecasting assistant. "
        "When given a list of numerical values, use the `generate_forecasts` tool to produce a forecast. "
        "Always ask the user for `prediction_length` and `quantile_levels` if not provided, do not assume or default any values. "
    ),
)

# ── App ──────────────────────────────────────────────────────────────────

# Create the AgentCore app
app = BedrockAgentCoreApp()


# Define the entrypoint of the AgentCore app
@app.entrypoint
async def invoke(payload: dict):
    """
    Stream agent events in response to a user message.

    Parameters:
    ===============================================================================
    payload: dict
        Request payload containing the user message under the "prompt" key.

    Yields:
    ===============================================================================
    dict
        Agent event dictionaries containing text chunks, tool use information,
        and lifecycle events emitted during agent execution.
    """
    stream = agent.stream_async(payload.get("prompt", ""))
    async for event in stream:
        yield event


# Run the AgentCore app
if __name__ == "__main__":
    app.run()
