# Deployment & infra cheat sheet

Reference doc for the k3d/Argo/Tailscale stack this project runs on: what each
piece is, how to rebuild it from scratch, and the commands you actually reach
for day to day. See also: [`pipelines.md`](pipelines.md) (kedro pipelines),
[`mlflow.md`](mlflow.md) (experiment tracking access), [`desktop-access.md`](desktop-access.md)
(SSH to the desktop).

## System map

```
Your laptop (zacpc)                    Desktop (desktop-fpi4cha), always-on
─────────────────────                  ──────────────────────────────────────
git push                                  Tailscale (mesh network, stable
  │                                       100.x.x.x IPs, no LAN/port-forward)
  ▼                                              │
GitHub Actions                                   │
  CI: pytest                                      ├─ k3d cluster "prod-311"
  CD: docker build → GHCR                          │   (k3s-in-docker, 1 node)
        ghcr.io/zacharycosenza/modeling:latest      │   └─ ns: argo
  │                                                 │       ├─ CronWorkflows (train/
  │  (image pulled by pods on next scheduled run)   │       │   inference/tweet-*)
  ▼                                                 │       └─ argo-server (UI)
  ...cluster pulls :latest...  ◄────────────────────┘             │
                                                                   ├─ port-forwarded to
                                                    kubectl apply ─┤   127.0.0.1:2746
                                                    (manual, see    │  (systemd unit)
                                                     "Deploying                │
                                                     changes" below)           ▼
                                                                   tailscale serve
                                                                   → https://desktop-fpi4cha.tailf82cf9.ts.net
                                                    mlflow ui process (separate,
                                                    see mlflow.md)
```

Pods write artifacts directly to the desktop's local `data/` folder via a
`hostPath` volume mount — no object storage, no PVC.

## Concepts

| Term | What it means here |
|---|---|
| **Image** | Frozen snapshot of the code + Python env, built by `docker build`, stored on GHCR. |
| **Container** | A running instance of an image. Many containers can run from one image. |
| **Cluster** | The set of machines Kubernetes manages together. Ours (`prod-311`) is a single-node k3d cluster running inside Docker on the desktop. |
| **Node** | One machine in the cluster — one Docker container standing in for a "real" machine. |
| **Namespace** | Logical partition inside a cluster (folders, basically). We use `argo` for everything; `argocd` would hold Argo CD if it were bootstrapped (it isn't — see below). |
| **Pod** | Smallest unit k8s runs — one container (or a few) sharing network/storage. Argo creates one per workflow run, deletes it when done. |
| **kubectl context** | Which cluster your `kubectl` commands target. Check with `kubectl config get-contexts` — easy to silently run commands against the wrong cluster if you have more than one. |
| **CronWorkflow** | An Argo-managed scheduled job definition (Argo v4 uses `spec.schedules`, an array — not the singular `spec.schedule`). |
| **Workflow** | One actual run of a CronWorkflow, each with its own pod and logs. |
| **serviceAccountName: argo** | The k8s identity workflow pods run as, needed so they can report status back to the Argo controller. Granted via `deploy/rbac.yaml`. |
| **Argo CD** | GitOps controller — watches a git path and applies it automatically. **Not installed on this cluster** (no `argocd` namespace exists). All deploys here are manual `kubectl apply`. |
| **Tailscale** | Private mesh VPN. Every enrolled device gets a stable `100.x.x.x` IP and a `*.ts.net` hostname, reachable from anywhere, no port-forwarding or dynamic-DNS needed. |
| **`tailscale serve`** | Reverse-proxies a local port to an HTTPS URL on your tailnet, with Tailscale's own TLS cert — no self-signed-cert warnings. |

## Reproducible setup (from scratch)

If the cluster is ever deleted, this rebuilds it. Run on the desktop unless noted.

```bash
# 1. Cluster, with local data/ mounted in
k3d cluster create prod-311 \
  --volume /home/cosenzac/code/project-311/data:/app/data@all

# 2. Argo Workflows (v4.0.8 — large CRDs need --server-side)
kubectl create namespace argo
kubectl apply -f https://github.com/argoproj/argo-workflows/releases/download/v4.0.8/install.yaml \
  --server-side --force-conflicts

# 3. RBAC — argo-cluster-role is missing `create` on workflowtaskresults by default
kubectl apply -f deploy/rbac.yaml

# 4. Twitter credentials (plain Secret — no sealed-secrets controller is running
#    despite kubeseal being installed; don't bother with kubeseal here)
kubectl create secret generic twitter-credentials -n argo \
  --from-literal=TWITTER_API_KEY=<...> \
  --from-literal=TWITTER_API_SECRET=<...> \
  --from-literal=TWITTER_ACCESS_TOKEN=<...> \
  --from-literal=TWITTER_ACCESS_TOKEN_SECRET=<...>

# 5. The CronWorkflows themselves
kubectl apply -f deploy/workflows/

# 6. (optional) Argo CD, if you want git-push-to-deploy instead of manual
#    kubectl apply — see "Argo CD" below. Not currently running.
```

**Argo Workflows UI over Tailscale** (browser access to workflow status +
pod logs, from any tailnet device) is its own multi-step setup — see
[Argo Workflows UI](#argo-workflows-ui-browser-access-over-tailscale) below.
It's not part of the cluster bootstrap above; it's an operability add-on
layered on top of an already-running `argo-server`.

## Deploying changes

**There is no auto-deploy for Kubernetes manifests.** CD only builds and
pushes the Docker image — it does not touch the cluster. The flow is:

```bash
git push origin main
# → CI runs tests
# → CD builds + pushes ghcr.io/zacharycosenza/modeling:latest (and :<sha>)
# → cluster picks up :latest on the NEXT scheduled CronWorkflow run
#   (imagePullPolicy: Always — no need to restart anything)
```

Code changes (anything under `src/`) reach the cluster automatically this
way, on the next scheduled run. **Changes to `deploy/*.yaml` do not** — those
need a manual step on the desktop:

```bash
ssh desktop
cd ~/code/project-311 && git pull
kubectl apply -f deploy/workflows/
# if a CronWorkflow was renamed or removed, the old one is orphaned —
# delete it explicitly:
kubectl get cronworkflows -n argo
kubectl delete cronworkflow <old-name> -n argo
```

The split architecture (`train`, `inference`, `tweet-summary`, `tweet-daily`)
has since been applied to the live cluster — a `train` pod run with the new
`securityContext.runAsUser: 1000` is what surfaced the Java/Ivy incident
documented under "Known gotchas" below, which only exists in the new YAML.
Worth a quick `kubectl get cronworkflows -n argo` to confirm the old `tweet`
CronWorkflow is actually gone next time you're in — it should have been
deleted as part of that rollout.

## CI/CD

- `.github/workflows/ci.yml` — runs `pytest tests/` on every push and PR.
- `.github/workflows/cd.yml` — after CI passes on `main`, builds the Docker
  image and pushes `ghcr.io/zacharycosenza/modeling:latest` + `:<sha>`.
  CronWorkflows use `:latest`.

**Gotchas:**
- GHCR image names must be lowercase; `github.repository_owner` preserves
  original casing — pipe it through `tr '[:upper:]' '[:lower:]'` into
  `$GITHUB_ENV` first.
- GHCR packages default to **private** — the cluster gets a 403 pulling
  `:latest` until you flip it to public on
  `github.com/ZacharyCosenza?tab=packages`. This has bitten us before
  (silently reverts to private occasionally — worth checking if a
  `CronWorkflow` pod is stuck in `ImagePullBackOff`).
- Integration tests hit real external APIs (Socrata, Open-Meteo) — skipped
  in CI via `@pytest.mark.skipif(os.environ.get("CI") == "true", ...)`.

## Deploy manifests

```
deploy/
  rbac.yaml              # Role + RoleBinding granting the argo ServiceAccount
                          # create/patch on workflowtaskresults
  application.yaml        # Argo CD Application (not bootstrapped — see below)
  workflows/
    train.yaml            # daily 7:00 AM ET — raw+target+features+modeling, registers model
    inference.yaml         # Mondays 8:00 AM ET — computes all-district predictions + SHAP reasons
    tweet_summary.yaml      # Mondays 8:15 AM ET — weekly summary tweet + district map
    tweet_daily.yaml         # Mon-Fri 8:20 AM ET — daily deep-dive tweet, one district/day
```

All CronWorkflows: `concurrencyPolicy: Forbid` (skip if a run's still going),
`imagePullPolicy: Always`, `serviceAccountName: argo`, `securityContext:
{runAsUser: 1000, runAsGroup: 1000}` (matches `cosenzac` on the desktop, so
pod-written files under the `hostPath` mount aren't `root`-owned), same
`hostPath` volume mount, and `env: [KEDRO_ENV, HOME]` set directly in each
workflow's own YAML (each file is self-contained — no shared ConfigMap; that
was tried and deliberately dropped in favor of keeping each workflow's full
config visible in its own file). The two tweet workflows additionally pull
`envFrom: secretRef: twitter-credentials`.

`inference` → `tweet_summary`/`tweet_daily` is a real dependency (both tweet
pipelines just read `inference_results.parquet`, never recompute) enforced
purely by schedule ordering, not a k8s-level `depends_on` — if `inference`
runs long or fails, the tweet jobs will read Monday's *previous* run's
artifact rather than blocking. Worth knowing if a tweet ever looks stale.

`HOME=/tmp` specifically exists to fix a real incident: the container image
has no `/etc/passwd` entry for uid 1000 (the `runAsUser` this pod runs as),
so without an explicit `HOME`, PySpark's `featurize_lags` step can't resolve
a home directory for its Ivy dependency-resolver setup and the JVM dies with
`[JAVA_GATEWAY_EXITED]` before Spark ever starts. See "Known gotchas" below.
Since it's set per-file rather than shared, adding a new env var means
editing all four `deploy/workflows/*.yaml` — deliberate tradeoff for now.

## RBAC

The built-in `argo-cluster-role` does **not** include `create` on
`workflowtaskresults` — pods can't report their status back without it.
`deploy/rbac.yaml` adds a Role + RoleBinding for the `argo` ServiceAccount.
The CronWorkflow must also explicitly set `serviceAccountName: argo` (the
default SA doesn't have this).

## Argo CD (not bootstrapped)

Would turn `git push` into the only deploy step for `deploy/`, watching the
repo and auto-`kubectl apply`-ing any diff (`selfHeal: true` — reverts manual
`kubectl edit`s back to whatever's in git). Currently not installed —
manifests are applied by hand (see "Deploying changes" above). To turn it on:

```bash
kubectl create namespace argocd
kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml \
  --server-side --force-conflicts
kubectl apply -f deploy/application.yaml   # one-time bootstrap; git push is enough after this
```

## Argo Workflows UI (browser access over Tailscale)

`argo-server` ships in-cluster only (`ClusterIP`), with an auth mode meant for
the CLI (`argo auth token` + a bearer header on every request). To browse
workflow status and pod logs from a normal browser, on any tailnet device,
without a token, three changes were made on the desktop:

1. **Auth mode relaxed.** `argo-server`'s deployment runs with
   `--auth-mode=server --secure=false` — plain HTTP internally, no
   per-request token. Safe specifically because Tailscale is the real access
   boundary here: nothing outside the tailnet can reach the pod either way.
   Its readiness probe was also patched from `HTTPS` to `HTTP` to match —
   otherwise the probe keeps expecting TLS and the rollout never goes ready.

   ```bash
   kubectl patch deploy -n argo argo-server --type=json -p='[
     {"op": "replace", "path": "/spec/template/spec/containers/0/args",
      "value": ["server", "--auth-mode=server", "--secure=false"]}
   ]'
   kubectl patch deploy -n argo argo-server --type=json -p='[
     {"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/httpGet/scheme",
      "value": "HTTP"}
   ]'
   ```

2. **Persistent port-forward.** A systemd *user* service on the desktop
   keeps `127.0.0.1:2746` forwarded to the `argo-server` service, restarting
   automatically if it dies (`Restart=always`) and surviving logout/reboot
   via `loginctl enable-linger cosenzac` (no root needed for your own
   account).

   `~/.config/systemd/user/argo-port-forward.service` on the desktop:
   ```ini
   [Unit]
   Description=Persistent kubectl port-forward for argo-server UI
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   Environment=KUBECONFIG=%h/.kube/config
   ExecStart=/snap/bin/kubectl port-forward -n argo svc/argo-server 2746:2746 --address=127.0.0.1
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=default.target
   ```

   ```bash
   systemctl --user daemon-reload
   systemctl --user enable --now argo-port-forward.service
   ```

3. **Tailnet exposure.** `tailscale serve` proxies the local port to a real
   HTTPS URL. Needed Serve enabled on the tailnet once (via the
   admin-console link the CLI prints the first time it's used) and the
   Tailscale "operator" set to your user so `serve` doesn't need `sudo`:

   ```bash
   sudo tailscale set --operator=cosenzac   # one-time
   tailscale serve --bg http://127.0.0.1:2746
   ```

**UI**: https://desktop-fpi4cha.tailf82cf9.ts.net

**Known limitation:** pod log retention is bounded by
`successfulJobsHistoryLimit`/`failedJobsHistoryLimit` on each CronWorkflow
(currently unset on all of them, i.e. k8s default), so old runs' logs
disappear once their pods are garbage-collected. If that becomes a real
problem, the natural next step is shipping logs to something like Grafana
Loki — not done, just flagging the extension point.

## Command cheat sheet

### Cluster / context

```bash
k3d cluster list                         # list clusters
k3d cluster delete prod-311              # delete cluster (data/ on disk survives)
kubectl config get-contexts              # list kubectl contexts
kubectl config use-context k3d-prod-311  # make sure you're pointed at the right one
```

### Workflows (day to day)

```bash
kubectl get cronworkflows -n argo                # what's scheduled
kubectl get workflows -n argo --sort-by=.metadata.creationTimestamp  # run history, oldest first
argo submit --from cronworkflow/train -n argo     # trigger a run manually, right now
argo get <workflow-name> -n argo                  # status of one run
argo logs <workflow-name> -n argo                 # stream/dump its logs
argo delete --completed -n argo                   # clean up finished runs
```

### Pods

```bash
kubectl get pods -n argo                          # list pods
kubectl describe pod <pod> -n argo                 # full detail — check here first for ImagePullBackOff etc.
kubectl logs <pod> -n argo -c main                  # container logs directly
kubectl delete pod <pod> -n argo                     # force-delete a stuck pod
```

### Secrets

```bash
kubectl get secrets -n argo
kubectl get secret twitter-credentials -n argo -o jsonpath='{.data}'  # base64 — pipe through base64 -d per key
```

### Tailscale

```bash
tailscale status                          # who's on the tailnet, IPs
tailscale serve status                    # what's currently being served
tailscale serve --bg http://127.0.0.1:2746  # (re-)start serving argo-server
tailscale serve --https=443 off           # stop serving
```

### Inspecting hostPath data (no PVC to worry about — it's just a folder)

```bash
ls -la data/prod/02_reporting/            # directly, no kubectl needed — it's a real dir on the desktop
```

## Known gotchas

| Problem | Fix |
|---|---|
| `spec.schedule unknown field` | Argo v4 uses `spec.schedules` (array), not `spec.schedule` |
| `repository name must be lowercase` | Pipe `github.repository_owner` through `tr '[:upper:]' '[:lower:]'` |
| `workflowtaskresults is forbidden` | `argo-cluster-role` missing `create` — apply `deploy/rbac.yaml` |
| `cannot change roleRef` | RoleBindings are immutable — delete and recreate, don't edit in place |
| Large CRD apply fails (262144 bytes) | Always use `--server-side --force-conflicts` for Argo installs |
| GHCR 403 / `ImagePullBackOff` | Package reverted to private — make it public again on `github.com/ZacharyCosenza?tab=packages` |
| `argo-server` readiness probe failing after an auth-mode change | Probe scheme (`HTTPS`/`HTTP`) has to match `--secure=<bool>` on the container — they're set independently and don't auto-sync |
| Manifest change in `deploy/` doesn't reach the cluster | There's no auto-sync (Argo CD isn't bootstrapped) — `kubectl apply -f deploy/workflows/` by hand on the desktop |
| `[JAVA_GATEWAY_EXITED]` in `featurize_lags` | No `/etc/passwd` entry for `runAsUser: 1000` in the image → Java can't resolve `user.home` → Spark's Ivy setup crashes on an invalid path. Fixed by adding `HOME=/tmp` to `env:` in each `deploy/workflows/*.yaml` |

## Current state

- [x] CI runs tests, CD builds + pushes image to GHCR on every merge to `main`
- [x] `train` CronWorkflow live, daily 7:00 AM ET
- [x] Artifacts written to `data/prod/02_reporting/` via `hostPath`
- [x] `twitter-credentials` Secret created (plain `kubectl create secret`, not sealed)
- [x] Argo Workflows UI exposed over Tailscale, no bearer token needed
- [x] Split `inference`/`tweet-summary`/`tweet-daily` CronWorkflows applied to the cluster
- [ ] `HOME=/tmp` (the fix for the Java/Ivy incident) added to each `deploy/workflows/*.yaml` but not yet `kubectl apply`'d to the cluster — next scheduled `train` run will still fail on `featurize_lags` until this ships
- [ ] Argo CD not bootstrapped — all cluster changes are manual `kubectl apply`
- [ ] `successfulJobsHistoryLimit`/`failedJobsHistoryLimit` unset on all CronWorkflows — old logs vanish once pods are GC'd
