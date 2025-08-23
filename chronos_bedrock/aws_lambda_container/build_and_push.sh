aws_account_id="<aws-account-id>"
region="<ecr-repository-region>"
algorithm_name="<ecr-repository-name>"

aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $aws_account_id.dkr.ecr.$region.amazonaws.com

docker build -t $algorithm_name .

docker tag $algorithm_name:latest $aws_account_id.dkr.ecr.$region.amazonaws.com/$algorithm_name:latest

docker push $aws_account_id.dkr.ecr.$region.amazonaws.com/$algorithm_name:latest