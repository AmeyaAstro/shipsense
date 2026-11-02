# Push to GitHub (one time)

Run in the VS Code terminal, inside the `shipsense_week1` folder (the repo root):

```bash
git init -b main
git add .
git status            # check: no data/, archive/ or *.db files listed
git commit -m "Week 1: EDA, SQL features, baselines, PyTorch model, tests"
git remote add origin https://github.com/AmeyaAstro/shipsense.git
git push -u origin main
```

If GitHub asks for a password, use a Personal Access Token (GitHub → Settings → Developer
settings → Tokens) or sign in through VS Code's Source Control panel.

If the repo already has a README from GitHub, run `git pull origin main --allow-unrelated-histories`
before `git push`.
