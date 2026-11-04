#!/usr/bin/env bash
# Deploy the ShipSense prediction API to AWS: model -> S3, image -> ECR, container -> Lambda + Function URL.
#
# Prereqs (see docs/aws_deploy.md): AWS CLI v2 configured (`aws sts get-caller-identity` works),
# Docker running, `make train` done (models/gradient_boosting.joblib exists).
# Safe to re-run: existing resources are reused and the function is updated in place.
# Undo everything with deploy/aws/teardown.sh.
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
NAME="shipsense"
BUCKET="${NAME}-models-${ACCOUNT_ID}-${REGION}"   # bucket names are global; account id keeps it unique
REPO="${NAME}-api"
ROLE="${NAME}-lambda-role"
FUNCTION="${NAME}-api"
MODEL_FILE="models/gradient_boosting.joblib"
MEMORY_MB=1024           # also scales CPU; sklearn + pandas start comfortably here
RESERVED_CONCURRENCY=2   # caps parallel copies, which caps the worst-case bill if the URL is abused

cd "$(dirname "$0")/../.."
[[ -f "$MODEL_FILE" ]] || { echo "missing $MODEL_FILE: run 'make train' first"; exit 1; }
step() { printf '\n== %s\n' "$1"; }

step "1/6 S3: upload the model (private bucket, versioned key)"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [[ "$REGION" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" >/dev/null
  else
    aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
fi
MODEL_HASH="$(shasum -a 256 "$MODEL_FILE" | cut -c1-12)"
MODEL_KEY="models/gradient_boosting-${MODEL_HASH}.joblib"   # new model = new key; old ones stay for rollback
aws s3 cp "$MODEL_FILE" "s3://${BUCKET}/${MODEL_KEY}" --only-show-errors
echo "s3://${BUCKET}/${MODEL_KEY}"

step "2/6 ECR: build the arm64 image and push it"
aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr put-lifecycle-policy --repository-name "$REPO" --region "$REGION" --lifecycle-policy-text \
  '{"rules":[{"rulePriority":1,"description":"keep last 3 images","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' >/dev/null
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
TAG="$(git rev-parse --short HEAD)"
IMAGE="${REGISTRY}/${REPO}:${TAG}"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
# --provenance=false: Lambda needs a plain image manifest, not the attestation index buildx adds by default
docker build --platform linux/arm64 --provenance=false -f deploy/aws/Dockerfile.lambda -t "$IMAGE" .
docker push "$IMAGE" >/dev/null
echo "$IMAGE"

step "3/6 IAM: role the function runs as (logs + read the model, nothing else)"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "waiting 10 s for the new role to propagate"; sleep 10
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name read-model --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::${BUCKET}/models/*\"}]}"
ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)"

step "4/6 Lambda: create or update the function"
ENV="Variables={SHIPSENSE_MODEL_S3_URI=s3://${BUCKET}/${MODEL_KEY}}"
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FUNCTION" --region "$REGION" --image-uri "$IMAGE" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION" --region "$REGION"
  aws lambda update-function-configuration --function-name "$FUNCTION" --region "$REGION" \
    --memory-size "$MEMORY_MB" --timeout 30 --environment "$ENV" >/dev/null
else
  aws lambda create-function --function-name "$FUNCTION" --region "$REGION" --package-type Image \
    --code "ImageUri=${IMAGE}" --role "$ROLE_ARN" --architectures arm64 \
    --memory-size "$MEMORY_MB" --timeout 30 --environment "$ENV" >/dev/null
fi
aws lambda wait function-updated-v2 --function-name "$FUNCTION" --region "$REGION"
# New accounts often have a total limit of 10, and AWS won't let you reserve below 10 unreserved;
# in that case the account limit itself already caps concurrency, so carry on.
aws lambda put-function-concurrency --function-name "$FUNCTION" --region "$REGION" \
  --reserved-concurrent-executions "$RESERVED_CONCURRENCY" >/dev/null 2>&1 \
  || echo "note: could not reserve concurrency (account limit too low); the account limit caps it instead"

step "5/6 Function URL: public HTTPS endpoint"
if ! aws lambda get-function-url-config --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FUNCTION" --region "$REGION" --auth-type NONE >/dev/null
  # Since Oct 2025 a public URL needs both statements; the flag limits the second to calls via the URL.
  aws lambda add-permission --function-name "$FUNCTION" --region "$REGION" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FUNCTION" --region "$REGION" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal "*" --invoked-via-function-url >/dev/null
fi
URL="$(aws lambda get-function-url-config --function-name "$FUNCTION" --region "$REGION" --query FunctionUrl --output text)"

step "6/6 Smoke test (first call is a cold start: image pull + model download, can take ~10-30 s)"
curl -sf --max-time 60 "${URL}health" && echo
curl -sf --max-time 30 -X POST "${URL}predict" -H "Content-Type: application/json" -d '{
  "order_ts": "2017-09-14T10:30:00", "shipping_mode": "Standard Class", "payment_type": "DEBIT",
  "market": "LATAM", "order_region": "Central America", "order_country": "México",
  "customer_segment": "Consumer", "main_department": "Fan Shop", "n_items": 2, "total_quantity": 3,
  "total_sales": 420.5, "avg_discount_rate": 0.1, "n_departments": 1}' && echo
printf '\nDeployed: %s\nDocs: %sdocs\nTear down: deploy/aws/teardown.sh\n' "$URL" "$URL"
