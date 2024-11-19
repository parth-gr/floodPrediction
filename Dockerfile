# Base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local files to the container
COPY . /app

# Create separate directories for each main file
RUN mkdir -p /app/src/main1 /app/src/main2

# Copy the main files to their respective directories
COPY src/main.py /app/src/main1/
COPY data/train.csv /app/src/main1/
COPY src/mainlr.py /app/src/main2/
COPY data/train.csv /app/src/main2/

# Install required system dependencies (if any)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if needed (for Flask, FastAPI, etc. if used in the future)
# EXPOSE 5000

# Set environment variables
ENV WANDB_API_KEY=your-wandb-api-key

# Create the entrypoint script directly in the Dockerfile
RUN echo '#!/bin/sh\n' \
    'if [ "$1" = "main1" ]; then\n' \
    '    python /app/src/main1/main.py\n' \
    'elif [ "$1" = "main2" ]; then\n' \
    '    python /app/src/main2/mainlr.py\n' \
    'else\n' \
    '    python /app/src/main1/main.py && python /app/src/main2/mainlr.py\n' \
    'fi\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Use ENTRYPOINT to run the script
ENTRYPOINT ["/app/entrypoint.sh"]

