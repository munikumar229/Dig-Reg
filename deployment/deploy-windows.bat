@echo off
REM 🐳 Dig-Reg Windows Deployment Script (Batch)
REM Simple batch file to deploy Dig-Reg on Windows

echo.
echo 🚀 Dig-Reg: Windows Deployment (Batch)
echo =====================================
echo.

REM Configuration
set DOCKER_USERNAME=munikumar229
set BACKEND_IMAGE=%DOCKER_USERNAME%/dig-reg-backend:latest
set FRONTEND_IMAGE=%DOCKER_USERNAME%/dig-reg-frontend:latest
set DEPLOY_DIR=dig-reg-deployment

REM Check if Docker is installed and running
echo [INFO] Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop for Windows.
    echo Download from: https://desktop.docker.com/win/main/amd64/Docker%%20Desktop%%20Installer.exe
    pause
    exit /b 1
)
echo [SUCCESS] Docker is installed

REM Check if Docker daemon is running
echo [INFO] Checking Docker daemon...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker daemon is not running. Please start Docker Desktop.
    echo Look for Docker Desktop in your system tray and start it.
    pause
    exit /b 1
)
echo [SUCCESS] Docker daemon is running

REM Check Docker Compose
echo [INFO] Checking Docker Compose...
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    docker compose version >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Docker Compose is not available. Please update Docker Desktop.
        pause
        exit /b 1
    ) else (
        set COMPOSE_CMD=docker compose
        echo [SUCCESS] Docker Compose (plugin) is available
    )
) else (
    set COMPOSE_CMD=docker-compose
    echo [SUCCESS] Docker Compose is available
)

REM Create deployment directory
echo [INFO] Creating deployment directory: %DEPLOY_DIR%
if exist "%DEPLOY_DIR%" (
    echo [WARNING] Directory %DEPLOY_DIR% already exists. Cleaning up...
    rmdir /s /q "%DEPLOY_DIR%" >nul 2>&1
)
mkdir "%DEPLOY_DIR%"
cd "%DEPLOY_DIR%"
echo [SUCCESS] Created directory: %DEPLOY_DIR%

REM Create docker-compose.yml
echo [INFO] Creating docker-compose.yml file...
(
echo version: '3.8'^

echo.^

echo services:^

echo   # FastAPI Backend Service^

echo   backend:^

echo     image: %BACKEND_IMAGE%^

echo     ports:^

echo       - "8000:8000"^

echo     environment:^

echo       - MLFLOW_TRACKING_URI=sqlite:///mlflow.db^

echo     volumes:^

echo       - mlruns:/app/mlruns^

echo       - models:/app/models^

echo     networks:^

echo       - app-network^

echo     healthcheck:^

echo       test: ["CMD", "curl", "-f", "http://localhost:8000/health"]^

echo       interval: 30s^

echo       timeout: 10s^

echo       retries: 3^

echo     restart: unless-stopped^

echo.^

echo   # Streamlit Frontend Service^

echo   frontend:^

echo     image: %FRONTEND_IMAGE%^

echo     ports:^

echo       - "8501:8501"^

echo     environment:^

echo       - BACKEND_URL=http://backend:8000^

echo     depends_on:^

echo       - backend^

echo     networks:^

echo       - app-network^

echo     restart: unless-stopped^

echo.^

echo networks:^

echo   app-network:^

echo     driver: bridge^

echo.^

echo volumes:^

echo   mlruns:^

echo   models:^

) > docker-compose.yml
echo [SUCCESS] Docker Compose file created

REM Pull images
echo [INFO] Pulling Docker images...
echo 📥 Pulling backend image: %BACKEND_IMAGE%
docker pull %BACKEND_IMAGE%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to pull backend image
    pause
    exit /b 1
)

echo 📥 Pulling frontend image: %FRONTEND_IMAGE%
docker pull %FRONTEND_IMAGE%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to pull frontend image
    pause
    exit /b 1
)
echo [SUCCESS] Both images pulled successfully

REM Stop any existing containers
echo [INFO] Stopping any existing Dig-Reg containers...
%COMPOSE_CMD% down --remove-orphans >nul 2>&1

REM Start services
echo [INFO] Starting Dig-Reg services...
%COMPOSE_CMD% up -d
if %errorlevel% neq 0 (
    echo [ERROR] Failed to start services
    pause
    exit /b 1
)

REM Wait for services
echo [INFO] Waiting for services to start...
timeout /t 5 /nobreak >nul

REM Check status
echo [INFO] Checking service status...
%COMPOSE_CMD% ps

REM Health checks
echo [INFO] Performing health checks...
echo 🔍 Backend health check...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] ✅ Backend is healthy
) else (
    echo [WARNING] ⚠️ Backend health check failed (may still be starting)
)

echo 🔍 Frontend availability check...
curl -s http://localhost:8501 >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] ✅ Frontend is accessible  
) else (
    echo [WARNING] ⚠️ Frontend not yet accessible (may still be starting)
)

REM Success message
echo.
echo 🎉 Deployment complete!
echo.
echo 📱 Access the application:
echo    Frontend (Streamlit): http://localhost:8501
echo    Backend API:          http://localhost:8000  
echo    API Documentation:    http://localhost:8000/docs
echo.
echo 🔧 Management commands:
echo    View logs:      %COMPOSE_CMD% logs -f
echo    Stop services:  %COMPOSE_CMD% down
echo    Restart:        %COMPOSE_CMD% restart
echo    Status:         %COMPOSE_CMD% ps
echo.
echo [SUCCESS] Dig-Reg is now running! 🚀

REM Ask to open browser
set /p OPEN_BROWSER="Would you like to open the application in your browser? (y/N): "
if /i "%OPEN_BROWSER%"=="y" (
    echo [INFO] Opening browser...
    start http://localhost:8501
)

echo.
echo Press any key to exit...
pause >nul

REM Return to parent directory
cd ..