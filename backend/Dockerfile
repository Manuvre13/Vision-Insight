# Use Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy all backend files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port where the FastAPI app will run
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
