# Base image with Python and CUDA support (if you need GPU support, change to a GPU-compatible image)
FROM pytorch/pytorch:1.13.1-cpu

# Set the working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY . .

# Set the Lambda runtime entry point
CMD ["handler.lambda_handler"]
