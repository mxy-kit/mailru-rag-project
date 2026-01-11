FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# ✅ 只安装一次：强制用 CPU torch 源（避免 nvidia-cuda-*）
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r requirements.txt

COPY . .

# 你的 ENTRYPOINT/ CMD 按你项目实际写
# 如果你离线推理脚本是 src/predict.py：
ENTRYPOINT ["python", "src/predict.py"]

