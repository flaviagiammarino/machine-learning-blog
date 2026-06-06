# AWS account ID
aws_account_id="<aws-account-id>"

# AWS region for all resources
region="<aws-region>"

# AgentCore Runtime name
agent_name="<agentcore-runtime-name>"

# ECR repository name for the container image
repository_name="<ecr-repository-name>"

# Secrets Manager secret ID for RDS credentials
secret_id="<secret-name>"

# RDS instance endpoint
db_host="<rds-db-host>"

# RDS database name
db_name="<rds-db-name>"

# IAM execution role for AgentCore Runtime
role_name="<agentcore-runtime-role>"

# Private subnet (routes through NAT Gateway for outbound internet access)
subnet_id="<private-subnet-id>"

# Security group (allows HTTPS outbound for AWS API calls and inbound from RDS)
security_group_id="<vpc-security-group>"

# Project setup
uv init --name postgres-agent --description "Strands Agent for Postgres Text-to-SQL" --python 3.13 --bare
uv lock
uv add strands-agents fastmcp postgres-mcp boto3 bedrock-agentcore

# ECR login and repository creation
aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $aws_account_id.dkr.ecr.$region.amazonaws.com
aws ecr create-repository --repository-name $repository_name --region $region

# Build and push container image
docker buildx build --platform linux/arm64 -t $aws_account_id.dkr.ecr.$region.amazonaws.com/$repository_name:latest --push .

# Deploy AgentCore Runtime
aws bedrock-agentcore-control create-agent-runtime \
  --agent-runtime-name "$agent_name" \
  --agent-runtime-artifact "{
    \"containerConfiguration\": {
      \"containerUri\": \"${aws_account_id}.dkr.ecr.${region}.amazonaws.com/${repository_name}:latest\"
    }
  }" \
  --network-configuration "{
    \"networkMode\": \"VPC\",
    \"networkModeConfig\": {
      \"subnets\": [\"${subnet_id}\"],
      \"securityGroups\": [\"${security_group_id}\"]
    }
  }" \
  --environment-variables "SECRET_ID=${secret_id},DB_HOST=${db_host},DB_NAME=${db_name}" \
  --role-arn "arn:aws:iam::${aws_account_id}:role/${role_name}" \
  --region "$region"