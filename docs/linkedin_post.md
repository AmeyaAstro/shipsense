# LinkedIn post draft

Edit it into your own voice before posting. Attach the demo video (or a screenshot), and put the
GitHub link in the post or the first comment.

---

I spent the last four weeks building ShipSense, an end-to-end ML project on 62,897 orders from the public DataCo supply-chain dataset. 📦

The question: when an order is placed, will it arrive late?

What I built:
• A gradient-boosting model (0.78 ROC-AUC on a time-based split), with features in SQL and re-implemented in PySpark with exact parity
• A FastAPI service in Docker, deployed to AWS Lambda (container image in ECR, model versioned in S3)
• MLflow tracking, GitHub Actions CI with 61 offline tests, and PSI drift monitoring
• An LLM ops copilot: RAG over policy docs plus tools that query the database and the model, running on a local 7B model

Three things I learned the hard way:

1️⃣ Leakage hides in plain sight. Many versions of this dataset report ~98% accuracy by using fields recorded *after* delivery. Without them, ~0.78 AUC is close to the real ceiling, and that's the honest number.

2️⃣ Small LLMs cut corners. In testing, the 7B model sometimes skipped its tools and invented a policy and a citation. Two guardrails helped: deterministic tool prefetching, and checking every citation against what was actually retrieved. On a 25-question eval, 24/25 answers were correct with them vs 23/25 without, and invented citations dropped from 3 to 1. Small set, so I treat that as directional.

3️⃣ Drift ≠ broken. Real monthly batches drifted massively on market and basket features, yet AUC held at 0.75–0.79, because lateness is driven by shipping mode, which didn't move. Alerting on every drifting feature would have fired every month, so the alerts are based on prediction drift and accuracy instead.

Bonus: my first Lambda deploy hit the 10-second init timeout. Lambda's CPU scales with memory, so doubling it cut the cold start from ~26 s to 7.7 s.

Code, reports and walkthroughs: github.com/AmeyaAstro/shipsense

I'm an MS CS (Data Science) student at UNC Charlotte, graduating May 2027, and looking for Data Science / ML Engineering roles. I'd love to hear feedback or connect!

#MachineLearning #MLOps #DataScience #AWS #LLM #RAG

---

**Before posting, check:**
- Every number still matches `docs/resume_bullets.md`.
- The policies are described as fictional if anyone asks.
