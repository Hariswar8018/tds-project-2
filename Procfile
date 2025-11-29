# ============================================
# File 1: Procfile (for Railway)
# ============================================
web: playwright install --with-deps chromium && uvicorn main:app --host 0.0.0.0 --port $PORT

# ============================================
# File 2: railway.json (Optional - Railway Config)
# ============================================
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "numReplicas": 1,
    "sleepApplication": false,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}

# ============================================
# File 3: nixpacks.toml (Railway Build Config)
# ============================================
[phases.setup]
nixPkgs = ["python39", "playwright-driver", "chromium"]

[phases.install]
cmds = [
  "pip install -r requirements.txt",
  "playwright install chromium"
]

[phases.build]
cmds = ["echo 'Build complete'"]

[start]
cmd = "uvicorn main:app --host 0.0.0.0 --port $PORT"
