# Use an official Python Alpine runtime as a parent image
FROM python:3.12-alpine

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apk add --no-cache gcc musl-dev linux-headers

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ ./src/

# Create a non-root user and switch to it
RUN adduser -D myuser
USER myuser

# Set the entrypoint
CMD ["python", "-m", "src.main"]