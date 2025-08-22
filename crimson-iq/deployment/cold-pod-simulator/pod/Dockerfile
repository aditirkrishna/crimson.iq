# Dockerfile for a sample Python service (e.g., pod, hospital, or blood bank)
# Save this as `Dockerfile` inside a directory like `pod/`

FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]