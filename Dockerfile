FROM eclipse-temurin:17-jre-jammy AS java
FROM python:3.12-slim

COPY --from=java /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="$JAVA_HOME/bin:$PATH"

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY conf/ conf/
RUN pip install --no-cache-dir -e .

# uid 1000 matches cosenzac on the desktop (see deploy/workflows/*.yaml
# securityContext.runAsUser) — needs a real passwd entry + home dir, or
# PySpark's JVM can't resolve a username/home and fails to start at all.
RUN useradd -u 1000 -m -s /bin/bash appuser
USER appuser

CMD ["kedro", "run"]
