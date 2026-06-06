import os
import json
import boto3
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp.client.stdio import stdio_client, StdioServerParameters
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# Create a Secrets Manager client and retrieve the database credentials
client = boto3.client("secretsmanager")
secret = client.get_secret_value(SecretId=os.environ["SECRET_ID"])

# Parse the secret string and extract the connection details
secret_string = json.loads(secret["SecretString"])
db_config = {
    "user": secret_string["username"],
    "password": secret_string["password"],
    "host": os.environ["DB_HOST"],
    "name": os.environ["DB_NAME"],
}

def create_postgres_client():
    # Configure the postgres-mcp server to run as a subprocess via uv
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "postgres-mcp"],
        env={"DATABASE_URI": f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['name']}?sslmode=verify-full&sslrootcert=/app/global-bundle.pem"}
    )
    # Return an MCP client that communicates with the server over stdio
    return MCPClient(lambda: stdio_client(server_params))

# Start the MCP client and keep it alive for the lifetime of the session
postgres_client = create_postgres_client()
postgres_client.__enter__()

# Retrieve the list of tools exposed by the postgres-mcp server
tools = postgres_client.list_tools_sync()

# Initialize the Strands agent with the postgres tools
agent = Agent(
    model="eu.anthropic.claude-sonnet-4-6",
    tools=tools,
    system_prompt=(
        "You are a text-to-SQL assistant with access to a PostgreSQL database. "
        "When answering questions, always inspect the database schema first to understand the available tables and columns. "
        "Generate accurate SQL queries based on the user's question and the actual schema. "
        "Return the query results in a clear, readable format. "
        "If a question cannot be answered from the available data, say so explicitly."
    )
)

# Create the AgentCore app
app = BedrockAgentCoreApp()

# Define the async streaming entrypoint
@app.entrypoint
async def invoke(payload: dict):
    # Stream agent events including text chunks, tool use, and results
    stream = agent.stream_async(payload.get("prompt", ""))
    async for event in stream:
        yield event

# Run the AgentCore app
if __name__ == "__main__":
    app.run()