
# Use official Python runtime as a base image
FROM python:3.11-slim

# Set environment vars
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install dependencies first (for caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Expose the port FastAPI will run on
EXPOSE 8080

# Command to run the app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
