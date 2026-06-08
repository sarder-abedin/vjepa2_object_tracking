FROM python:3.10-slim

# System libraries needed by OpenCV's GUI (cv2.imshow) and webcam capture.
# This is a Linux-based image: it builds and runs the same way through
# Docker on Linux, macOS, and Windows (Docker Desktop / WSL2) hosts.
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

COPY tracker_webcam.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "tracker_webcam.py"]
