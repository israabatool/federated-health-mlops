FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Expose port for serving
EXPOSE 5000

# Start the service
CMD ["python", "src/server.py"]

