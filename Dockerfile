FROM python:3.11-slim

WORKDIR /app

# system deps (optional but helpful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN python -m pip install --upgrade pip && `
    python -m pip install --no-cache-dir `
      --index-url https://download.pytorch.org/whl/cpu `
      --extra-index-url https://pypi.org/simple `
      -r requirements.txt


# copy source code
COPY . .

# IMPORTANT: we pull DVC artifacts at runtime (or you can do it at build time)
# If you want build-time pull, you need dvc + remote config inside image.
# We'll do runtime pull in entrypoint script to keep image build simple.

ENTRYPOINT ["python", "src/predict.py"]
