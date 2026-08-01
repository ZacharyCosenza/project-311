# Deployment walkthrough: weekly retrain + daily tweet

Scope for this pass: get the model retraining weekly and a tweet posting daily,
running on Kubernetes via Argo Workflows, deployed with zero manual `kubectl apply`
via Argo CD. The dashboard is explicitly out of scope here — it's a natural
follow-on addition to the same `/deploy` folder once this is working.

## 0. Assumptions

- You have `kubectl` pointed at a real Kubernetes cluster (cloud-managed or
  self-hosted; the commands below are cloud-agnostic).
- Images are pushed to GHCR (`ghcr.io`), since it's free and tied to your
  existing GitHub account — no separate registry signup.
- Model artifacts hand off between the weekly train job and daily tweet job via
  an in-cluster **MinIO** bucket (S3-compatible), so you don't need an external
  cloud storage account to get started. Swap for real S3/GCS later if you want —
  the client code doesn't change, just the endpoint/credentials.
- Twitter posting logic doesn't exist in this repo yet — this guide flags where
  it needs to go, with a skeleton, but you'll need your own approved Twitter
  Developer app and API v2 keys (from developer.twitter.com) before step 7 works
  end to end.

## 1. Architecture at a glance

```
push to main
   -> GitHub Actions: run tests, build image, push to ghcr.io/<you>/modeling:<sha>
   -> (Argo CD Image Updater notices the new tag, bumps it in /deploy)
   -> Argo CD notices the git diff, syncs the cluster
   -> CronWorkflow "weekly-train" (Mondays): runs `modeling.main train`,
      writes model.pkl to MinIO
   -> CronWorkflow "daily-tweet" (daily): loads latest model.pkl from MinIO,
      runs inference on the current week, posts a tweet
```

The only manual step, ever, is `git push`. Everything after that is automatic.

## 2. Prerequisites to install once

- [Argo CLI](https://github.com/argoproj/argo-workflows/releases) (optional,
  useful for manually triggering/watching workflow runs while testing)
- A GHCR personal access token with `write:packages` scope, OR just use the
  built-in `GITHUB_TOKEN` in Actions (simpler, no extra token to manage)
- A Twitter Developer account with API v2 read/write access on the app you'll
  post from

## 3. Containerize the app

Add `Dockerfile` at the repo root:

```dockerfile
FROM eclipse-temurin:17-jre-jammy AS java
FROM python:3.12-slim

COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "modeling.main"]
```

(PySpark needs a JVM at runtime — that's what the multi-stage copy of the JRE
is for, same reason the CI workflow installs Java for the test job.)

## 4. CI: build and push the image on merge

Extend `.github/workflows/ci.yml` with a second job that only runs after tests
pass, and only on pushes to `main` (not PRs):

```yaml
  build-and-push:
    needs: test
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/modeling:${{ github.sha }}
```

## 5. Install Argo Workflows and Argo CD on the cluster

```bash
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/latest/download/install.yaml

kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```

Both are one-time installs — you don't redo this per deploy.

## 6. Model artifact storage (MinIO)

Simplest path: install via Helm into its own namespace.

```bash
helm repo add minio https://charts.min.io/
kubectl create namespace storage
helm install minio minio/minio -n storage \
  --set rootUser=admin,rootPassword=<choose-a-password>,mode=standalone
```

Then create a bucket (port-forward the MinIO console or use `mc`):

```bash
kubectl port-forward -n storage svc/minio 9000:9000
mc alias set local http://localhost:9000 admin <choose-a-password>
mc mb local/modeling-artifacts
```

`main.py`'s train mode currently writes `model.pkl` to local disk
(`data/02_reporting/`) — it'll need a small change to also push that file to
`s3://modeling-artifacts/model.pkl` (via `boto3`, pointed at the MinIO
endpoint) so the daily-tweet job can fetch the latest one. Flagging this as a
follow-up code change, not something this doc does for you.

## 7. Secrets, the GitOps-safe way

Committing a plain `Secret` YAML to git leaks the credential into git history
forever. Use **Sealed Secrets** instead — it encrypts client-side, so the
committed blob is only decryptable by the cluster's controller.

```bash
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/latest/download/controller.yaml
# install the matching `kubeseal` CLI locally, then:
kubectl create secret generic twitter-api-keys \
  --from-literal=api-key=<...> --from-literal=api-secret=<...> \
  --from-literal=access-token=<...> --from-literal=access-secret=<...> \
  --dry-run=client -o yaml | kubeseal -o yaml > deploy/secrets/twitter-sealed.yaml
```

Repeat the same pattern for MinIO access credentials. The resulting
`*-sealed.yaml` files are safe to commit — that's the whole point.

## 8. Repo layout for GitOps

```
deploy/
  application.yaml          # the Argo CD Application itself
  workflows/
    weekly-train.yaml
    daily-tweet.yaml
  secrets/
    twitter-sealed.yaml
    minio-sealed.yaml
```

## 9. The two CronWorkflows

`deploy/workflows/weekly-train.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: weekly-train
  namespace: argo
spec:
  schedule: "0 6 * * 1"       # Monday 06:00 UTC
  timezone: "America/New_York"
  concurrencyPolicy: Forbid
  workflowSpec:
    entrypoint: train
    templates:
      - name: train
        container:
          image: ghcr.io/<you>/modeling:latest   # bumped automatically, see step 10
          command: ["python", "-m", "modeling.main", "train"]
          envFrom:
            - secretRef: {name: minio-credentials}
```

`deploy/workflows/daily-tweet.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: daily-tweet
  namespace: argo
spec:
  schedule: "0 14 * * *"      # 14:00 UTC daily
  concurrencyPolicy: Forbid
  workflowSpec:
    entrypoint: tweet
    templates:
      - name: tweet
        container:
          image: ghcr.io/<you>/modeling:latest
          command: ["python", "-m", "modeling.post_tweet"]   # doesn't exist yet, see below
          envFrom:
            - secretRef: {name: minio-credentials}
            - secretRef: {name: twitter-api-keys}
```

`modeling.post_tweet` doesn't exist in the codebase yet. It needs to: pull the
latest `model.pkl` from MinIO, pull/build this week's feature row(s), call
`pipeline_modeling.inference`, format the result into tweet text, and post via
the Twitter API (`tweepy` is the usual client). The actual tweet *content*
(which board, what phrasing) is a product decision worth designing separately
— happy to build this out once you're ready, it's a small, self-contained
script.

## 10. Wire up the image tag automatically

Rather than hand-editing the `image:` tag in both CronWorkflow YAMLs on every
release, install **Argo CD Image Updater** — it watches the GHCR repository
and rewrites the tag in git itself when a new image lands, so CI never needs
write access back into `/deploy`. Annotate both workflow manifests:

```yaml
metadata:
  annotations:
    argocd-image-updater.argoproj.io/image-list: modeling=ghcr.io/<you>/modeling
    argocd-image-updater.argoproj.io/modeling.update-strategy: latest
```

## 11. The Argo CD Application (the one bootstrap step)

`deploy/application.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: modeling
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/<you>/project-311.git
    targetRevision: main
    path: deploy
  destination:
    server: https://kubernetes.default.svc
    namespace: argo
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

Apply this **once**, by hand — it's the single bootstrap step that turns on
GitOps for everything else:

```bash
kubectl apply -f deploy/application.yaml
```

From this point on, every future change is `git push` only. Argo CD will
even revert manual `kubectl` edits made directly against the cluster, since
`selfHeal: true` means git is the only source of truth.

## 12. Verify it works

```bash
argocd app get modeling                 # confirm it's synced
argo submit --from cronworkflow/weekly-train -n argo --watch   # trigger a real run now, don't wait for Monday
```

## What's deferred

- The dashboard — a `Deployment` + `Service` (+ `Ingress` if it should be
  public) dropped into `deploy/` alongside the two CronWorkflows, once you
  want it.
- Swapping in-cluster MinIO for real S3/GCS, if you outgrow it.
- Swapping Sealed Secrets for External Secrets Operator, if you move to a
  real secrets manager (Vault, AWS/GCP Secrets Manager) later.

## Open items before this is fully live

1. Write `modeling/post_tweet.py` (needs the tweet-format decision).
2. Push/pull `model.pkl` to/from MinIO from `main.py` (currently local-disk only).
3. Get Twitter Developer API v2 credentials.
4. Confirm cluster specifics (provider, `kubectl` context) if not already set up.
