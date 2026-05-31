@echo off
echo Fallyx Gateway Local Development Runner
echo ======================================

echo Building and starting the container with docker-compose...
cd ..\..
docker-compose -f local/local-build/docker-compose.yml up --build -d

echo.
echo Container started successfully!
echo.
echo To view logs:
echo   docker logs -f fallyx-gateway-local
echo.
echo To stop the container:
echo   docker-compose -f local/local-build/docker-compose.yml down
