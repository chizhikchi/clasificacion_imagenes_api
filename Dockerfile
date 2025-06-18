FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model and application files
COPY efficientnet_v2_model.pth .
COPY app.py .
# Copiar conjunto de validación
COPY validation/ /app/validation/

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"] 