# --- kill anything stuck on port 5000 (check first — don't kill blindly if you have it open elsewhere) ---
ps aux | grep mlflow
kill $(lsof -ti :5000)   # or: fuser -k 5000/tcp

# --- copy the latest mlflow.db down from the desktop ---
mkdir -p data/prod/02_reporting
scp desktop:~/code/project-311/data/prod/02_reporting/mlflow.db data/prod/02_reporting/mlflow.db

# --- view it locally ---
.venv/bin/mlflow ui --backend-store-uri sqlite:///data/prod/02_reporting/mlflow.db --port 5000

# --- better: run it once on the desktop itself, then just browse over Tailscale forever ---
ssh desktop
cd ~/code/project-311
.venv/bin/mlflow server --backend-store-uri sqlite:///data/prod/02_reporting/mlflow.db --host 0.0.0.0 --port 5000
# then from any machine on the tailnet:
#   http://<desktop-tailscale-ip>:5000