# Week 4 walkthrough — monitor, demo, deploy

Budget about 90 minutes. Open each file as you read it, and run the commands yourself.

## 1. The big picture
Weeks 1–3 built a model, a service and a copilot. Week 4 answers three questions that come after
"it works on my laptop":

| Question | Answer | Files |
|---|---|---|
| "Is the model still OK on new data?" | PSI drift monitoring | `src/shipsense/drift.py`, `reports/drift/` |
| "Can a non-engineer see it working?" | Streamlit demo UI | `app/streamlit_app.py` |
| "Can it run in the cloud?" | AWS Lambda + S3 + ECR | `deploy/aws/`, `docs/aws_deploy.md` |

## 2. Drift monitoring — `drift.py`
**The idea:** the model learned from 2015–2016 orders. If new orders look different, its scores can
silently degrade. We compare each new month with the training data.

**PSI (Population Stability Index)**, the standard measure in credit risk:
1. **Bin the training data:** deciles for numbers, one bin per category for text. Unseen categories go
   into an "other" bin.
2. **Compute each bin's share** in training (ref%) and in the new batch (cur%).
3. **PSI = Σ (cur% − ref%) × ln(cur% / ref%).** A tiny `EPS` avoids `log(0)` when a bin is empty.
4. **Read it:** < 0.10 stable · 0.10–0.25 watch · > 0.25 act.

**Three layers:**
- **Feature PSI:** has the input data changed?
- **Prediction PSI:** have the model's scores changed?
- **Performance:** batch ROC-AUC once real outcomes arrive. In production they arrive days later.

**What we found on real data** (`make drift ARGS=--summary`):
- **Feature PSI is enormous every month (4.5–12).** DataCo rotates markets in monthly blocks (e.g.
  Feb–May 2017 are 100% LATAM), and from Oct 2017 orders become single-item with much lower sales.
- **But monthly AUC stays 0.754–0.792**, against 0.782 on the test set, and shipping-mode PSI never
  exceeds 0.006. Lateness is driven by shipping mode, which didn't move.
- **So the first alert rule ("act if any feature drifts") fired every single month while the model was
  fine.** That's **alert fatigue**: a monitor that always alarms gets ignored.
- **The fix:** the overall status comes from **prediction PSI**, and is forced to "act" if batch
  **AUC drops more than 0.05** below 0.782. Feature PSI is shown as the *diagnosis*. Result: 12 months
  stable, Oct 2017 "watch" (the month the product mix changed), and the synthetic shift "act".

**The synthetic scenario** (`--simulate-shift`): 40% of a month's Standard Class orders are relabelled
First Class, and it is clearly marked as not real. Prediction PSI jumps to 0.37 and AUC falls to 0.64,
so the monitor alerts. This proves the alarm can fire.

**Tests (`test_drift.py`):**
- PSI matches a value worked out by hand.
- Identical samples give 0.
- A shifted mean and unseen categories give a high PSI.
- The alert rule says "stable" for no change and alerts for the synthetic shift.

**Why there's no scheduled GitHub Action:** CI never has the dataset, so a scheduled job could only
re-run tests on fake data. That would *look* like monitoring without monitoring anything. In
production, schedule `drift.py` where the data lives (e.g. EventBridge Scheduler → a Lambda or ECS task).

## 3. Streamlit demo — `app/streamlit_app.py`
- **Three tabs:** a risk form (→ `POST /predict`), ask-the-copilot (→ `POST /ask`, showing sources, the
  tool trace and any invented citations), and the drift summary.
- **Thin client:** the UI calls the API over HTTP and never loads the model, so it shows exactly what any
  other system would get. Point it at AWS with `SHIPSENSE_API_URL=<url> make demo`.
- **Friendly failure:** if the API is down, `call_api` returns an error message instead of crashing.
- **A data gotcha:** the training data has `"Health and Beauty "` *with a trailing space*. Sending it
  without the space would be an unseen category to the model, so the dropdown keeps exact values and
  only trims them for display.
- **Test:** Streamlit's `AppTest` renders the app with the API down and checks the error appears. It was
  also driven live against the real API: 38.6% for a Standard Class order, 99.9% "HIGH RISK" for First
  Class, and a correct copilot answer.

## 4. AWS deployment — `deploy/aws/`, `docs/aws_deploy.md`
**Choosing the service** (checked against current AWS facts, not memory):
- **App Runner:** closed to new customers since 30 April 2026.
- **ECS Fargate:** bills 24/7 even with no traffic.
- **Lambda, chosen:** pay per request, scales to zero, has an always-free allowance, and a Function URL
  gives free HTTPS.

**How the pieces fit:**
1. **S3:** a private bucket holds the model under a key containing its hash
   (`gradient_boosting-9a54d461cf83.joblib`). A new model is a new key, and old ones stay for rollback.
2. **ECR:** a private registry for the image. It scans images on push and keeps only the last 3.
3. **`Dockerfile.lambda`:** the same FastAPI app plus the **AWS Lambda Web Adapter**, which turns Lambda
   events into ordinary HTTP requests for uvicorn. It adds `boto3`, and has **no model inside**.
4. **`api.py`:** if `SHIPSENSE_MODEL_S3_URI` is set, the model is downloaded to `/tmp` (Lambda's only
   writable folder) at startup. The URI is validated *before* importing boto3, so bad config fails fast.
5. **IAM role:** least privilege. The function may write logs and read `models/*` from one bucket,
   nothing else.
6. **Function URL:** public HTTPS. Since October 2025 a public URL needs *two* permissions; the second
   is restricted to calls that come through the URL.
7. **`deploy.sh` / `teardown.sh`:** re-runnable, and each step tolerates "already exists" or "already
   gone". Teardown asks for confirmation and verifies everything is deleted.

**What happened on the real deploy** (5 Nov 2026):
- It took 2 min 12 s.
- The smoke test on AWS returned the **same model hash and the same prediction (0.4078)** as the Mac.
- **The cold-start problem:**
  - At 1,024 MB, the logs showed `INIT_REPORT ... Status: timeout`. Startup (importing pandas and
    scikit-learn, downloading the model) exceeded Lambda's 10 s init limit, so the first request took
    ~26 s.
  - Lambda gives CPU in proportion to memory, so raising it to 2,048 MB fixed it: init 7.2 s, first
    request 7.7 s. RAM use was only ~330 MB either way. Now the default in `deploy.sh`.
- **Warm requests:** median 65 ms, p95 73 ms from a laptop.
- **Reserved concurrency of 2 was refused.** New accounts have a limit of 10, and AWS keeps at least 10
  unreserved. The account limit is the cap; the docs state the worst-case cost and rely on a Zero-spend
  budget alert plus teardown.

**Cost:**
- A demo is well under $1/month: Lambda stays inside its free allowance, and the 173 MB image costs
  about $0.02/month in ECR.
- Risk: the public URL could be abused. Protections: the budget alert, teardown after the demo, or
  `AWS_IAM` auth for a private URL.

**Why the URL is not in the README:** it stops working after teardown, and a public link invites random
traffic to a pay-per-use endpoint.

## 5. How to run Week 4
```bash
make drift ARGS=--summary                      # monthly table -> reports/drift/summary.md
make drift MONTH=2018-01 ARGS=--simulate-shift # synthetic shift report
make serve   # terminal 1
make demo    # terminal 2 -> http://localhost:8501
deploy/aws/deploy.sh      # needs AWS CLI + Docker; read docs/aws_deploy.md first
deploy/aws/teardown.sh    # when finished
make test                 # 61 offline tests
```

## 6. Resume bullets this week unlocks (truthful, measured)
- Deployed the FastAPI model service to AWS Lambda as an arm64 container: ECR image, model artifact
  versioned in S3, least-privilege IAM role, HTTPS Function URL, scripted deploy/teardown. Diagnosed a
  cold-start init timeout from CloudWatch logs and cut first-request latency from ~26 s to 7.7 s by
  right-sizing memory; warm p95 73 ms.
- Built PSI-based drift monitoring (feature, prediction and performance drift) and found that real
  monthly batches drifted heavily (feature PSI 4.5–12) while AUC held at 0.754–0.792. Redesigned
  alerting around prediction drift and AUC drop to avoid alert fatigue (12 of 13 months stable; a
  synthetic shift correctly flagged).
- Built a Streamlit demo (risk scoring, RAG copilot with sources and tool trace, drift dashboard) as a
  thin client over the API, with 61 offline tests in CI-gated GitHub Actions.

## 7. Interview questions — answer out loud, then check the hint
1. **What is PSI, and how do you compute it?**
   *Hint:* bin on training data, compare bin shares, Σ (cur − ref) ln(cur/ref); < 0.1 / 0.1–0.25 / > 0.25.
2. **Why bin on the training data's deciles, not the new batch's?**
   *Hint:* the reference defines "normal"; bins from the new batch would move with the drift.
3. **Your features drifted massively but AUC didn't drop. How?**
   *Hint:* the drifted features matter little to the prediction; shipping mode drives it and didn't move.
4. **What is alert fatigue, and how did you avoid it?**
   *Hint:* an always-firing alarm gets ignored; alert on prediction drift and AUC drop, and use feature
   PSI to diagnose.
5. **Data drift vs concept drift?**
   *Hint:* data drift is when inputs change; concept drift is when the input→label relationship changes.
   You need labels (AUC) to see concept drift.
6. **Why Lambda instead of ECS Fargate or App Runner here?**
   *Hint:* pay-per-request and scale to zero suit a demo; App Runner is closed to new customers; Fargate
   bills 24/7.
7. **What is a cold start, and how did you fix yours?**
   *Hint:* starting a new execution environment. Init hit the 10 s limit at 1 GB; Lambda's CPU scales
   with memory, and 2 GB → 7.7 s.
8. **Why keep the model in S3 instead of in the image?**
   *Hint:* models change more often than code; the hashed key gives versioning and rollback without
   rebuilding.
9. **What permissions does the Lambda role have, and why so few?**
   *Hint:* logs + `s3:GetObject` on `models/*` in one bucket. Least privilege limits the damage if the
   function is compromised.
10. **How do you stop a public endpoint from running up a bill?**
    *Hint:* a budget alert, a concurrency cap (blocked here by the account floor), IAM auth, teardown;
    API Gateway throttling is the paid option.
11. **Why is the Streamlit app a thin client?**
    *Hint:* one source of truth for predictions; the UI can point at local or AWS without changes.
12. **Why no scheduled drift job in GitHub Actions?**
    *Hint:* CI has no data; a job on fake data would only pretend to monitor. Schedule it where the data
    lives.

## 8. Your to-do
1. Record the demo using `docs/demo_video_script.md` while the AWS deploy is live.
2. **Then run `deploy/aws/teardown.sh`** and delete the `shipsense-deployer` access key in IAM.
3. Add two screenshots to `docs/images/` and uncomment them in the README.
4. Copy the bullets you want from `docs/resume_bullets.md`; post using `docs/linkedin_post.md`.
5. Answer the 12 questions above out loud.
