# Base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local files to the container
COPY . /app

COPY data /app/data

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

# Command to run the script
CMD ["python", "src/main.py"]
