agentcore configure \
    --entrypoint agent.py \
    --name forecasting_agent \
    --deployment-type direct_code_deploy \
    --runtime PYTHON_3_12 \
    --requirements-file requirements.txt \
    --non-interactive

agentcore launch
