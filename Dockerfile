FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    libffi-dev \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust + Cargo
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Create app dir
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --prefer-binary -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["python", "app.py"]