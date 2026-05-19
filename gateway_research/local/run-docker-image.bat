@echo off
echo Fallyx Gateway Docker Image Runner
echo ================================

REM Check if the image file exists
if not exist "%~1" (
  echo Error: Please provide the path to the Docker image file.
  echo Usage: %0 path\to\fallyx-gateway-docker-XXXXXXX.tar.gz
  exit /b 1
)

echo Loading Docker image from %~1...
echo This may take a few minutes depending on the image size.

REM Extract and load the Docker image using Docker's built-in functionality
docker load -i "%~1"

REM Get the image tag from the filename
for %%F in ("%~1") do set "FILENAME=%%~nF"
set "IMAGE_TAG=%FILENAME:~20,7%"

echo Image loaded successfully.
echo Starting container with tag: fallyx-gateway:%IMAGE_TAG%

REM Run the container using Docker Compose
echo Creating docker-compose-temp.yml file...
(
echo version: '3'
echo services:
echo   fallyx-gateway:
echo     image: fallyx-gateway:%IMAGE_TAG%
echo     container_name: fallyx-gateway
echo     privileged: true
echo     volumes:
echo       - /sys/fs/cgroup:/sys/fs/cgroup:ro
echo     ports:
echo       - "22:22"
echo       - "80:80"
echo       - "443:443"
echo       - "1883:1883"
echo       - "8080:8080"
echo     restart: unless-stopped
echo     environment:
echo       - PYTHONUNBUFFERED=1
echo     command: ["/sbin/init"]
) > docker-compose-temp.yml

echo Starting container with docker-compose...
docker-compose -f docker-compose-temp.yml up -d

echo.
echo Container started successfully!
echo.
echo You can access the container with:
echo   docker exec -it fallyx-gateway /bin/bash
echo.
echo To check the gateway service status:
echo   docker exec -it fallyx-gateway systemctl status gateway.service
echo.
echo To view logs:
echo   docker exec -it fallyx-gateway journalctl -u gateway.service
echo.
echo To stop the container:
echo   docker-compose -f docker-compose-temp.yml down
