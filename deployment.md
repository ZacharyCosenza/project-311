# Deployment: weekly retrain + daily tweet

## Concepts

A map of all the tools and how they connect — read this once before touching any commands.

### Docker: images and containers

A **Docker image** is a frozen snapshot of an environment: your code, Python version,
dependencies, all baked in. Think of it as a recipe or a class definition.

A **container** is a running instance of that image — a live process in an isolated box.
Like an object instantiated from a class. You can run many containers from the same image.

```
Image (ghcr.io/zacharycosenza/modeling:latest)
  └─ Container A  (running train job)
  └─ Container B  (another run, same image)
```

Images are built by `docker build`, stored in a **registry** (GHCR in our case),
and pulled to wherever they run (your laptop, a cloud server, a Kubernetes pod).

### Kubernetes: clusters, nodes, namespaces, pods

**Kubernetes (k8s)** is a system for running and scheduling containers at scale.
Instead of running `docker run` yourself, you declare what you want and Kubernetes
makes it happen.

- **Cluster** — a group of machines (real or virtual) that Kubernetes manages together.
  We use **k3d** to create a lightweight cluster (`prod-311`) that runs inside Docker
  on your desktop. One Docker container = one fake "machine" in the cluster.

- **Node** — one machine in the cluster. k3d clusters have one or more nodes,
  each a Docker container.

- **Namespace** — a logical partition inside a cluster. Like folders in a filesystem.
  We use `argo` for workflows and `argocd` for Argo CD. Keeps unrelated things from
  colliding.

- **Pod** — the smallest unit Kubernetes runs. One pod = one (or a few) containers
  that share a network and storage. When Argo runs your training job, it creates a pod,
  waits for it to finish, then deletes it.

- **kubectl** — the CLI to talk to Kubernetes. It always targets whichever cluster is
  set as your current **context** (`kubectl config get-contexts`). This is important
  if you have multiple clusters — commands go to the active context only.

```
Cluster (prod-311, running in Docker via k3d)
  └─ Namespace: argo
      └─ Pod: train-abc123  (your modeling container, created on schedule)
  └─ Namespace: argocd
      └─ Pod: argocd-server-...  (the Argo CD UI/controller)
```

### Argo Workflows: CronWorkflow, Workflow, templates

**Argo Workflows** is a Kubernetes-native job scheduler. You define jobs as YAML
and Argo creates pods to run them on schedule.

- **CronWorkflow** — a scheduled job definition, like a cron job but managed by
  Kubernetes. Argo v4 uses `spec.schedules` (an array), not `spec.schedule`.

- **Workflow** — one actual run of a CronWorkflow. Each run gets its own name
  (`train-abc123`), its own pod, and its own logs.

- **Template** — the unit of work inside a workflow. For us it's a single container
  step that runs `python -m modeling.main train`.

- **serviceAccountName** — which Kubernetes identity the pod runs as. Pods need
  permission to talk to the Kubernetes API (to report their status back to Argo).
  We use the `argo` service account, which has those permissions via `deploy/rbac.yaml`.

```
CronWorkflow "train"  (definition, always exists)
  └─ Workflow "train-abc123"  (one run, created hourly)
      └─ Pod "train-abc123"   (the actual container)
```

### Argo CD: GitOps

**Argo CD** watches a git repo and keeps the cluster in sync with it. Instead of
running `kubectl apply` manually after every change, you `git push` and Argo CD
applies the diff automatically.

- **Application** — an Argo CD resource that says "watch this repo path and sync
  it to this cluster namespace." We have one: `deploy/application.yaml`.
- **selfHeal: true** — if someone manually edits the cluster with `kubectl`, Argo CD
  reverts it. Git is the single source of truth.
- **Bootstrap** — Argo CD itself must be applied once by hand (`kubectl apply -f
  deploy/application.yaml`). After that, all future changes go through git.

### CI/CD: GitHub Actions

**CI** (Continuous Integration) and **CD** (Continuous Deployment) are automated
pipelines that run when you push code.

- **CI** (`ci.yml`) — runs tests. Catches broken code before it ships.
  Runs on every push and every pull request.

- **CD** (`cd.yml`) — builds the Docker image and pushes it to GHCR.
  Only runs after CI passes on `main`. Triggered via `workflow_run`.

- **GHCR** (GitHub Container Registry) — GitHub's free image registry.
  Images are stored at `ghcr.io/<owner>/<name>:<tag>`.
  By default packages are private — you must make them public for the cluster to pull.

```
git push
  -> CI runs tests
      -> (if pass) CD builds image, pushes ghcr.io/zacharycosenza/modeling:latest
          -> k3d cluster pulls :latest on next CronWorkflow run
```

### How it all connects in this project

```
Your code (src/)
  -> Dockerfile bakes it into an image
      -> GitHub Actions (CD) pushes image to GHCR
          -> k3d cluster (prod-311) pulls the image
              -> Argo Workflows runs it as a pod on schedule
                  -> Pod writes artifacts to /app/data (= your local data/ folder)
```

Argo CD (not yet bootstrapped) would sit between GitHub and kubectl — watching
`deploy/` in git and auto-applying any YAML changes to the cluster.

---

## Architecture

```
push to main
  -> GitHub Actions CI: run tests (skips integration tests in CI)
  -> GitHub Actions CD: build image, push to ghcr.io/zacharycosenza/modeling:latest + :<sha>
  -> CronWorkflow "train" (weekly): runs modeling.main train, writes artifacts to local data/prod/
  -> CronWorkflow "tweet" (daily, not yet implemented): reads model, posts tweet
```

Local Kubernetes cluster (`prod-311`) runs on k3d (k3s in Docker). Artifacts are written
directly to your local `data/` folder via a hostPath volume mount — no object storage needed.

## Cluster setup (one-time, if cluster is deleted)

```bash
# Create cluster with local data/ folder mounted inside
k3d cluster create prod-311 --volume /home/cosenzac/code/project-311/data:/app/data@all

# Install Argo Workflows (v4.0.8 — note: uses spec.schedules array, not spec.schedule)
kubectl create namespace argo
kubectl apply -f https://github.com/argoproj/argo-workflows/releases/download/v4.0.8/install.yaml \
  --server-side --force-conflicts

# Install Argo CD
kubectl create namespace argocd
kubectl apply -n argocd \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml \
  --server-side --force-conflicts

# Apply all deploy/ manifests
kubectl apply -f deploy/rbac.yaml
kubectl apply -f deploy/workflows/train.yaml
```

## CI/CD

Split into two workflows:

- `.github/workflows/ci.yml` — runs tests on every push and PR
- `.github/workflows/cd.yml` — builds and pushes Docker image to GHCR after CI passes on main

**Gotchas:**
- GHCR image names must be all-lowercase. `github.repository_owner` preserves original casing.
  Fix: `echo '${{ github.repository_owner }}' | tr '[:upper:]' '[:lower:]'` into `$GITHUB_ENV`.
- CD pushes both `:latest` and `:<sha>` tags. The CronWorkflow uses `:latest`.
- Integration tests hit real external APIs — skip them in CI with:
  `@pytest.mark.skipif(os.environ.get("CI") == "true", reason="requires external API access")`

## Deploy manifests

```
deploy/
  rbac.yaml              # Role + RoleBinding for argo service account
  application.yaml       # Argo CD Application (bootstrap once, then git push is enough)
  workflows/
    train.yaml           # CronWorkflow: retrain + register model in mlflow
    tweet.yaml            # CronWorkflow: daily forecast + tweet (champion model promoted manually in mlflow)
```

## RBAC

Argo Workflows pods need permission to write `workflowtaskresults`. The built-in
`argo-cluster-role` does NOT include `create` on that resource — you must add it.
`deploy/rbac.yaml` adds a custom Role + RoleBinding for the `argo` service account.

The CronWorkflow must also set `serviceAccountName: argo` (not the default SA).

## Argo CD bootstrap (not yet done)

Run once to turn on GitOps — after this, `git push` is the only deploy step:

```bash
kubectl apply -f deploy/application.yaml
```

## Useful commands

### Cluster

```bash
k3d cluster list                         # list clusters
k3d cluster delete prod-311              # delete cluster
kubectl config get-contexts              # list kubectl contexts
kubectl config use-context k3d-prod-311  # switch to prod-311
```

### Workflows

```bash
# Trigger a run manually
argo submit --from cronworkflow/train -n argo

# Watch a running workflow
argo get <workflow-name> -n argo

# Stream logs
argo logs <workflow-name> -n argo

# List all workflow runs
kubectl get workflows -n argo

# List CronWorkflows
kubectl get cronworkflows -n argo

# Delete a workflow run
argo delete <workflow-name> -n argo

# Delete all completed/failed runs
argo delete --completed -n argo
```

### Pods

```bash
kubectl get pods -n argo                 # list pods
kubectl describe pod <pod> -n argo       # full pod details
kubectl logs <pod> -n argo -c main       # container logs
kubectl delete pod <pod> -n argo         # delete a pod
```

### Inspect PVC contents (if not using hostPath)

```bash
kubectl run inspect --image=busybox --restart=Never -n argo \
  --overrides='{"spec":{"volumes":[{"name":"data","persistentVolumeClaim":{"claimName":"modeling-data"}}],"containers":[{"name":"inspect","image":"busybox","command":["ls","-la","/data/prod/02_reporting"],"volumeMounts":[{"name":"data","mountPath":"/data"}]}]}}'
kubectl wait pod/inspect -n argo --for=condition=Ready --timeout=30s || true
kubectl logs inspect -n argo
kubectl delete pod inspect -n argo
```

### Argo CD

```bash
kubectl get applications -n argocd       # list apps
kubectl describe application modeling -n argocd
```

## Current state

- [x] CI runs tests, CD builds and pushes image to GHCR
- [x] CronWorkflow `train` runs hourly (change to `0 6 * * 1` for production)
- [x] Artifacts written to local `data/prod/02_reporting/` via hostPath volume
- [ ] Argo CD not yet bootstrapped (`kubectl apply -f deploy/application.yaml`)
- [ ] Tweet workflow not yet implemented (`modeling/post_tweet.py`)
- [ ] Twitter credentials not yet sealed

## Pending: tweet workflow

`modeling/post_tweet.py` needs to: read `model.pkl` from `data/prod/02_reporting/`,
run inference on the current week's features, format as tweet text, post via tweepy.

Twitter credentials go in `.secrets/twitter.env` (gitignored), sealed for cluster use:

```bash
kubectl create secret generic twitter-api-keys \
  --from-literal=api-key=<...> \
  --from-literal=api-secret=<...> \
  --from-literal=access-token=<...> \
  --from-literal=access-secret=<...> \
  --dry-run=client -o yaml | kubeseal -o yaml > deploy/secrets/twitter-sealed.yaml
```

## Known gotchas

| Problem | Fix |
|---|---|
| `spec.schedule unknown field` | Argo v4 uses `spec.schedules` (array) |
| `repository name must be lowercase` | Pipe `github.repository_owner` through `tr '[:upper:]' '[:lower:]'` |
| `workflowtaskresults is forbidden` | `argo-cluster-role` missing `create` — apply `deploy/rbac.yaml` |
| `cannot change roleRef` | RoleBindings are immutable — delete and recreate |
| `--server-side` flag | Large CRDs exceed client-side annotation limit (262144 bytes) — always use `--server-side --force-conflicts` for Argo installs |
| GHCR 403 on image pull | Package is private by default — make it public on github.com/ZacharyCosenza?tab=packages |
