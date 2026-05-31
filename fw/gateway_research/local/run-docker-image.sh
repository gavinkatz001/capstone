#!/bin/bash

echo "Fallyx Gateway Docker Image Runner"
echo "================================"

# Check if the image file exists
if [ ! -f "$1" ]; then
  echo "Error: Please provide the path to the Docker image file."
  echo "Usage: $0 path/to/fallyx-gateway-docker-XXXXXXX.tar.gz"
  exit 1
fi

echo "Loading Docker image from $1..."
echo "This may take a few minutes depending on the image size."

# Extract and load the Docker image using Docker's built-in functionality
docker load -i "$1"

# Get the image tag from the filename
FILENAME=$(basename "$1")
IMAGE_TAG=${FILENAME:20:7}

echo "Image loaded successfully."
echo "Starting container with tag: fallyx-gateway:$IMAGE_TAG"

# Run the container using Docker Compose
echo "Creating docker-compose-temp.yml file..."
cat > docker-compose-temp.yml << EOF
version: '3'
services:
  fallyx-gateway:
    image: fallyx-gateway:$IMAGE_TAG
    container_name: fallyx-gateway
    privileged: true
    volumes:
      - /sys/fs/cgroup:/sys/fs/cgroup:ro
    ports:
      - "22:22"
      - "80:80"
      - "443:443"
      - "1883:1883"
      - "8080:8080"
    restart: unless-stopped
    environment:
      - PYTHONUNBUFFERED=1
    command: ["/sbin/init"]
EOF

echo "Starting container with docker-compose..."
docker-compose -f docker-compose-temp.yml up -d

echo
echo "Container started successfully!"
echo
echo "You can access the container with:"
echo "  docker exec -it fallyx-gateway /bin/bash"
echo
echo "To check the gateway service status:"
echo "  docker exec -it fallyx-gateway systemctl status gateway.service"
echo
echo "To view logs:"
echo "  docker exec -it fallyx-gateway journalctl -u gateway.service"
echo
echo "To stop the container:"
echo "  docker-compose -f docker-compose-temp.yml down"
