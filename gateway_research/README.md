# CAPTSONE GATEWAY PRELIMINARY RESEARCH
# FOLDER STILL UNDER CONSTRUCTION
# FINAL DECISION:WE WILL INCORPERATE GATEWAY USE INTO OUR DESIGN

## Overview

Fallyx-Gateway is a firmware project for the Orange Pi Zero 2 W that serves as a bridge between BLE-connected sensors and WiFi-connected cloud services. The gateway device collects data from sensors via BLE, processes it, and transmits it to cloud endpoints via WiFi.

## Key Features

- **BLE Communication**: Continuous scanning for and connecting to sensors, receiving data
- **Data Processing**: Creating structured JSON files from sensor data
- **WiFi Transmission**: Sending data to specific endpoints
- **Bidirectional Communication**: Receiving commands from server and relaying to sensors

## Hardware Requirements

The firmware is designed for the Orange Pi Zero 2 W, which was selected based on the following criteria:
- Support for BLE (including LE Coded PHY Config)
- Dipole antenna port (SMA or U-FL)
- WiFi connectivity
- Ability to handle 7-8 concurrent connections
- USB port for speaker and microphone
- Approximately 1 GB of RAM
- Sustainable supply chain

## Documentation

For more detailed information about the project, please refer to the documentation in the `docs` directory:

- [Project Overview](docs/01-project-overview.md)
- [Implementation Plan](docs/02-implementation-plan.md)
- [Data Flow Architecture](docs/03-data-flow-architecture.md)
- [Implementation Considerations](docs/08-implementation-considerations.md)
- [Image Building Guide](docs/image-building-guide.md)

## Building and Testing the Gateway

### Building the Gateway Image

This repository includes a GitHub Actions workflow that automates the process of building a custom Armbian image for the Orange Pi Zero 2 W with the gateway firmware pre-installed. The workflow is defined in `.github/workflows/build-image.yml`.

The workflow produces two main artifacts:
1. An Armbian image that can be flashed to an SD card for the Orange Pi Zero 2 W
2. A Docker image that can be used for local testing without physical hardware

For detailed instructions on how to build and use these images, please refer to the [Image Building Guide](docs/image-building-guide.md).

### Local Testing with Docker

The GitHub Actions workflow creates a Docker image that can be used for local testing without physical hardware. This is particularly useful for:
- Rapid development iterations
- Testing basic functionality
- CI/CD pipeline integration
- Development environments without access to physical hardware

#### Using Docker Compose (Recommended)

The repository includes a `docker-compose.yml` file for easy testing:

```bash
# Load the Docker image
gunzip -c fallyx-gateway-docker-<commit-sha>.tar.gz | docker load

# Update the image tag in docker-compose.yml if needed

# Run the container
docker-compose up -d

# Access the container
docker exec -it fallyx-gateway /bin/bash
```

#### Using Docker Directly

Alternatively, you can use Docker commands directly:

```bash
# Load the Docker image
gunzip -c fallyx-gateway-docker-<commit-sha>.tar.gz | docker load

# Run the container
docker run --name fallyx-gateway --privileged -d fallyx-gateway:<commit-sha>

# Access the container
docker exec -it fallyx-gateway /bin/bash
```

Note that while Docker testing is convenient, it has limitations for hardware-dependent features. For complete testing, especially of BLE and WiFi functionality, you should use the actual Orange Pi Zero 2 W hardware.

For more detailed instructions on using the Docker image, see the [Image Building Guide](docs/image-building-guide.md).

## Getting Started

1. Clone this repository
2. Review the documentation to understand the project architecture
3. Build the gateway image using the GitHub Actions workflow or manually following the guide
4. Flash the image to an SD card and insert it into the Orange Pi Zero 2 W
5. Power on the device and it will automatically start the gateway firmware


