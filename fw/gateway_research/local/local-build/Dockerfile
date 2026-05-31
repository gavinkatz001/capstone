FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bluetooth \
    bluez \
    libbluetooth-dev \
    libglib2.0-dev \
    sqlite3 \
    libsqlite3-dev \
    libasound2-dev \
    portaudio19-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/fallyx-gateway

# Copy gateway code
COPY gateway/ /opt/fallyx-gateway/gateway/

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/fallyx-gateway/gateway/requirements.txt

# Set hostname
RUN echo 'fallyx-gateway' > /etc/hostname

# Expose ports
EXPOSE 1883 8080

# Set entrypoint to run the gateway directly
CMD ["python", "-m", "gateway.main"]
