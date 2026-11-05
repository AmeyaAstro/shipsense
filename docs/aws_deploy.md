# Deploying ShipSense to AWS (Lambda container + S3 + ECR)

What gets deployed: the **prediction API** (`/health`, `/predict`, `/predict/batch`, `/docs`).
The copilot (`/ask`) is **not** deployed: it needs a local LLM (Ollama) and a 4.7 GB model, which
don't fit a free-tier setup. Run it locally.

```
models/gradient_boosting.joblib --upload--> S3 (private bucket, key includes the model hash)
deploy/aws/Dockerfile.lambda --build arm64--> ECR (private repo, last 3 images kept)
Lambda (container image, arm64, 2 GB) --reads model at cold start--> S3
Function URL (public HTTPS) --> Lambda Web Adapter --> uvicorn --> FastAPI
```

## Why this design
- **Not App Runner:** closed to new customers since 30 April 2026 (AWS recommends ECS Express Mode
  instead).
- **Not ECS Fargate:** runs (and bills) 24/7 even with no traffic.
- **Lambda:**
  - Bills per request and per millisecond.
  - Has an always-free monthly allowance (1M requests, 400,000 GB-seconds).
  - Scales to zero when idle.
  - The **Lambda Web Adapter** runs our ordinary uvicorn server unchanged, so the same FastAPI code
    runs locally, in Docker and on Lambda.
  - A **Function URL** gives HTTPS with no API Gateway charge.
- **Model in S3, not baked into the image:** a new model is an S3 upload plus a config change, not a
  new image. Old model files stay in S3 for rollback.
- **arm64:** your Mac builds arm64 natively, and Lambda's arm64 (Graviton) is about 20% cheaper per
  GB-second than x86.

## Cost (Nov 2026 list prices; confirm on aws.amazon.com/pricing before relying on them)

| Item | Size / use | Expected cost |
|---|---|---|
| Lambda | demo traffic (hundreds to thousands of requests) | **$0**, inside the always-free allowance |
| ECR storage | 173 MB compressed image (measured), up to 3 kept | ≈ $0.02–0.05 / month |
| S3 | 0.6 MB model | < $0.01 / month |
| CloudWatch Logs | a few MB | $0 (free allowance) |
| **Total for a demo left up for a month** | | **well under $1** |

**Worst case, and why it's capped.** The URL is public. If someone hammered it 24/7:
- With reserved concurrency 2 at 2 GB, that's up to roughly **$120–180/month**.
- New accounts have a limit of 10 concurrent executions, and AWS won't reserve below that floor
  (this happened on the real deploy). Then the account limit of 10 is the only cap, and the bill
  could be up to 5× that.

That's why the budget alert and teardown below matter.

Protections, in order of importance:
1. **Tear down after recording the demo** (`deploy/aws/teardown.sh`).
2. **Set a $1 budget alert** (step A3 below).
3. If you don't need a public URL, switch it to `--auth-type AWS_IAM`.

Accounts created after 15 July 2025 start on AWS's free plan with credits. Check
**Billing → Free tier / Plan** to see which plan yours is on.

## A. One-time setup (you click; about 15 minutes)
1. **Sign in** to the AWS console as the root user. Turn on **MFA** for root
   (top right → Security credentials → Assign MFA device).
2. **Pick a region:** use **US East (N. Virginia), `us-east-1`** (top-right region menu). The
   scripts default to it.
3. **Budget alert:** Billing and Cost Management → Budgets → Create budget → *Use a template* →
   **Zero spend budget**, or a *Monthly cost budget* of **$1**. Enter your email → Create.
4. **A deploy user (don't use root for the CLI):**
   1. IAM → Users → Create user → name `shipsense-deployer`.
   2. Attach policies directly: **AdministratorAccess**. This is a personal sandbox; in a company
      you'd scope it to S3, ECR, Lambda, IAM role creation and Logs.
   3. Open the user → Security credentials → **Create access key** → *Command Line Interface (CLI)*.
   4. Keep the window open for step B2. **Never paste these keys into chat, code or a file in the
      repo.**
5. **Install the AWS CLI:** download https://awscli.amazonaws.com/AWSCLIV2.pkg, double-click, and
   choose **"Install for all users of this computer"** (asks for your Mac password, and puts `aws` on
   your PATH automatically). Open a new terminal and check that `aws --version` works.

## B. Configure the CLI (you type; keys stay on your Mac)
1. **Enter your keys:**
   ```bash
   aws configure          # paste the access key + secret, region us-east-1, output json
   aws sts get-caller-identity   # should print your account id and the shipsense-deployer ARN
   ```
   The keys are saved in `~/.aws/credentials`, outside the repo.

## C. Deploy (about 5 minutes)
Docker Desktop must be running.
```bash
make train                 # if models/ is empty
deploy/aws/deploy.sh       # S3 -> ECR -> IAM role -> Lambda -> Function URL -> smoke test
```
- It prints the URL. Open `<URL>docs` for the interactive API.
- The **first request is a cold start**: Lambda starts the container, imports pandas and scikit-learn,
  and downloads the model. Measured at 2 GB: about 7–8 s. Later requests are fast until the function
  has been idle a while.
- Point the Streamlit demo at it:
  `SHIPSENSE_API_URL=<URL without trailing slash> make demo` (the copilot tab will say it's unavailable).
- Re-running `deploy.sh` updates everything in place. A retrained model gets a new S3 key and a new
  image tag.

## Measured on the real deploy (5 Nov 2026, us-east-1, arm64)
| What | Result |
|---|---|
| `deploy.sh` end to end (S3 → ECR → IAM → Lambda → URL → smoke test) | 2 min 12 s |
| Smoke test on AWS | same model hash (`9a54d461cf83`) and same prediction (0.4078) as local |
| Cold start at **1024 MB** | init hit Lambda's 10 s limit (`Status: timeout`), retried; first request ≈ 26 s |
| Cold start at **2048 MB** | init 7.2 s, first request 7.7 s (now the default) |
| Warm `/predict`, 100 calls from a laptop over the internet | median 65 ms, p95 73 ms |
| `/predict/batch` with 1,000 orders (1024 MB) | median 350 ms |
| Memory actually used | ~330 MB |
| Reserved concurrency 2 | refused: new account limit is 10 (the account limit caps it instead) |

The lesson: Lambda gives CPU in proportion to memory, so a CPU-bound startup (importing pandas and
scikit-learn) can need more "memory" even though RAM use is only ~330 MB.

## D. Tear down (after the demo)
```bash
deploy/aws/teardown.sh     # asks for confirmation, deletes Lambda, logs, ECR repo, S3 bucket, IAM role
```
Then, in the console:
- Check **Billing → Bills** the next day.
- Optionally delete the `shipsense-deployer` access key (IAM → Users → Security credentials).

## What each AWS piece is, in one line (for interviews)
| Service | Role here |
|---|---|
| **S3** | object storage: holds versioned model files |
| **ECR** | private Docker registry: holds the API image |
| **IAM role** | the function's identity: may write logs and read `models/*` from one bucket, nothing else (least privilege) |
| **Lambda** | runs the container on demand, scales to zero, bills per ms |
| **Function URL** | public HTTPS endpoint for the function |
| **CloudWatch Logs** | where uvicorn's logs land |
