FROM python:3.10-slim

# Declare working directory
WORKDIR /app

# Installs the necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements.txt file to take advantage of the Docker cache
COPY requirements.txt .

# Installs Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Exposes the port used by Uvicorn
EXPOSE 8000

# Command to launch application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
