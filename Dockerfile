FROM ultralytics/ultralytics:latest-arm64

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt