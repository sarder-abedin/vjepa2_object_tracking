FROM python:3.10-slim

# System libraries required by OpenCV's GUI (cv2.imshow) and webcam capture
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libgtk2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -U pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Default: V-JEPA2 SSV2 action recognition + YOLO/OWL-ViT detection demo
CMD ["python", "vjepa2_ssv2_action_prediction_webcam.py"]
