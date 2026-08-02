FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8080

# Installed before the source is copied so that editing code does not invalidate
# the dependency layer. numpy, scipy, scikit-learn and pyarrow are all wheels on
# this platform, so no compiler is needed.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY static ./static
COPY sql ./sql
COPY configs ./configs

# Neither the parquet files nor the loader are in the image. The container reads
# from Postgres, and loading is a one-off run against the database from wherever
# the data was generated, which is also the only place pyarrow is needed.

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8080

# One worker. Requests are long-lived SSE streams that spend their time waiting
# on the model rather than on CPU, and clustering already uses several cores
# through BLAS, so extra workers would multiply memory for no throughput.
CMD ["sh", "-c", "python -m uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8080} --timeout-keep-alive 75"]
