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
