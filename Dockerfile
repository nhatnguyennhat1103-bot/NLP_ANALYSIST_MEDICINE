FROM python:3.12-slim-bookworm

# Đặt thư mục làm việc
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết (quan trọng cho torch/transformers)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy và cài đặt thư viện trước để tận dụng Docker Cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào container
COPY . .

# Lệnh chạy ứng dụng
CMD ["python", "main.py"]