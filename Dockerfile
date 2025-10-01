# Use slim Python 3.10
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Hugging Face cache to writable directory
ENV TRANSFORMERS_CACHE=/app/cache
RUN mkdir -p /app/cache

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Uvicorn port
EXPOSE 7860

# Start command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
