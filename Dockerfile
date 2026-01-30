FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ocean Protocol uses the $ALGO environment variable to point to the algorithm file
# Using shell form so that $ALGO gets expanded at runtime
ENTRYPOINT python $ALGO
