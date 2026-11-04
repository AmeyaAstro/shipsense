#!/usr/bin/env bash
# Delete everything deploy.sh created, so nothing keeps costing money.
# Order matters: remove what uses a resource before the resource itself.
# Each step tolerates "already gone", so it is safe to re-run.
set -uo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)" || exit 1
NAME="shipsense"
BUCKET="${NAME}-models-${ACCOUNT_ID}-${REGION}"
REPO="${NAME}-api"
ROLE="${NAME}-lambda-role"
FUNCTION="${NAME}-api"

read -r -p "Delete ShipSense's Lambda, ECR repo '$REPO', S3 bucket '$BUCKET' and IAM role '$ROLE' in $REGION? [y/N] " ok
[[ "$ok" == "y" ]] || { echo "aborted"; exit 1; }

echo "== Lambda function + URL";  aws lambda delete-function --function-name "$FUNCTION" --region "$REGION" 2>/dev/null
echo "== CloudWatch logs";        aws logs delete-log-group --log-group-name "/aws/lambda/${FUNCTION}" --region "$REGION" 2>/dev/null
echo "== ECR repo + images";      aws ecr delete-repository --repository-name "$REPO" --region "$REGION" --force >/dev/null 2>&1
echo "== S3 bucket + model";      aws s3 rb "s3://${BUCKET}" --force >/dev/null 2>&1
echo "== IAM role"
aws iam delete-role-policy --role-name "$ROLE" --policy-name read-model 2>/dev/null
aws iam detach-role-policy --role-name "$ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null
aws iam delete-role --role-name "$ROLE" 2>/dev/null

echo; echo "Check nothing is left (each should say 'not found' / be empty):"
aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1 && echo "  ! lambda still exists" || echo "  lambda: gone"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 && echo "  ! ECR repo still exists" || echo "  ECR: gone"
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null && echo "  ! bucket still exists" || echo "  S3: gone"
aws iam get-role --role-name "$ROLE" >/dev/null 2>&1 && echo "  ! role still exists" || echo "  IAM role: gone"
