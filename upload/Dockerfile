# Use an official Python runtime as a parent image
FROM python:3.12-slim


# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies using pip 
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ ./src/

# Run as a non-root user
RUN useradd -m myuser
USER myuser


# Set the entrypoint
CMD ["python", "-m", "src.main"]