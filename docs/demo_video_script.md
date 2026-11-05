# Demo video script (~3 minutes)

Record with macOS **Cmd + Shift + 5 → Record Selected Portion**, and use a quiet room. Upload the video
**unlisted** to YouTube or Loom, and link it from the README and LinkedIn.

## Before you record (10 min)
1. Docker Desktop is running.
2. The AWS deploy is still up (`deploy/aws/deploy.sh` if you tore it down).
3. **Warm up Lambda** so the viewer doesn't watch a 7 s cold start:
   `curl -s <URL>health`
4. Terminal 1: `make serve`. Terminal 2: `make demo`. Ollama is running.
5. **Warm up the copilot** so the first answer isn't slow on camera: ask one question in the UI first.
6. Have these open in browser tabs:
   - the Streamlit UI (http://localhost:8501)
   - `<URL>docs` (AWS)
   - the GitHub repo
   - `reports/drift/summary.md` on GitHub.
7. Close notifications. Zoom the browser to 125%.

## Script

| Time | Show | Say (roughly; don't read word for word) |
|---|---|---|
| 0:00–0:20 | GitHub README (top table) | "ShipSense predicts, when an order is placed, whether it'll arrive late. It's served as an API on AWS, monitored for drift, and has an LLM copilot for operations questions. Everything is measured, tested in CI, and on GitHub." |
| 0:20–0:45 | README results table | "62,897 orders from a public supply-chain dataset, time-based split. Gradient boosting gets 0.78 ROC-AUC. Projects that claim 98% use post-delivery fields that *are* the label; I excluded those, and a test enforces it." |
| 0:45–1:15 | Streamlit → Risk tab: Standard Class → **Predict** (≈39%), switch to First Class → **Predict** (≈99.9%, HIGH RISK) | "The UI calls the API, so it's exactly what any system would get. First Class is almost always late in this data, because its one-day schedule is almost never met, and the model learned that." |
| 1:15–1:35 | AWS `<URL>docs` → POST /predict → Execute | "This is the same API running on AWS Lambda: container in ECR, model in S3. Same model hash, same prediction as my laptop. When I first deployed, cold starts hit Lambda's 10-second init limit; doubling memory gave more CPU and brought it to under 8 seconds." |
| 1:35–2:25 | Streamlit → Copilot: ask *"Is order 62477 high risk, and what should we do for high-risk orders?"* Open **Tool calls** and **Sources** | "The copilot runs a local 7B model. It retrieves the relevant policy, looks up the order, and runs the risk model, and you can see every tool call. In testing, the small model sometimes skipped tools and invented citations, so the agent prefetches tools and flags any citation it didn't actually retrieve. On a 25-question eval it got 24 right." |
| 2:25–2:50 | Streamlit → Drift tab | "Drift monitoring: real months drifted massively on market and basket features, but AUC held steady because shipping mode drives lateness. So alerts are based on prediction drift and accuracy, not every feature; otherwise it would alarm every month." |
| 2:50–3:00 | GitHub Actions page (green runs) | "61 tests run offline on every push. Links are in the description. Thanks for watching." |

## After recording
1. Watch it once and cut dead air. Add captions (YouTube auto-captions plus a quick fix).
2. Take two screenshots (Risk tab, Copilot answer with tool calls), save them as
   `docs/images/risk_tab.png` and `docs/images/copilot_tab.png`, and uncomment them in the README.
3. **Run `deploy/aws/teardown.sh`.** The video is your proof; the live URL doesn't need to stay up.
