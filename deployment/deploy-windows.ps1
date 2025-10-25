# 🐳 Dig-Reg Windows Deployment Script
# PowerShell script to deploy Dig-Reg on Windows

param(
    [string]$DockerUsername = "munikumar229"
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Colors for output
$Green = [ConsoleColor]::Green
$Red = [ConsoleColor]::Red
$Yellow = [ConsoleColor]::Yellow
$Blue = [ConsoleColor]::Blue
$White = [ConsoleColor]::White

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Yellow
}

# Banner
Write-Host "🚀 Dig-Reg: Windows Deployment Script" -ForegroundColor $Blue
Write-Host "======================================" -ForegroundColor $Blue
Write-Host ""

# Configuration
$BackendImage = "${DockerUsername}/dig-reg-backend:latest"
$FrontendImage = "${DockerUsername}/dig-reg-frontend:latest"
$DeploymentDir = "dig-reg-deployment"

try {
    # Check if Docker is installed
    Write-Info "Checking Docker installation..."
    try {
        $dockerVersion = docker --version
        Write-Success "Docker is installed: $dockerVersion"
    }
    catch {
        Write-Error "Docker is not installed. Please install Docker Desktop for Windows."
        Write-Host "Download from: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe" -ForegroundColor $Yellow
        exit 1
    }

    # Check if Docker is running
    Write-Info "Checking Docker daemon..."
    try {
        docker info | Out-Null
        Write-Success "Docker daemon is running"
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop."
        Write-Host "Look for Docker Desktop in your system tray and start it." -ForegroundColor $Yellow
        exit 1
    }

    # Check if Docker Compose is available
    Write-Info "Checking Docker Compose availability..."
    try {
        $composeVersion = docker-compose --version
        Write-Success "Docker Compose is available: $composeVersion"
        $ComposeCmd = "docker-compose"
    }
    catch {
        try {
            docker compose version | Out-Null
            Write-Success "Docker Compose (plugin) is available"
            $ComposeCmd = "docker compose"
        }
        catch {
            Write-Error "Docker Compose is not available. Please update Docker Desktop."
            exit 1
        }
    }

    # Create deployment directory
    Write-Info "Creating deployment directory: $DeploymentDir"
    if (Test-Path $DeploymentDir) {
        Write-Warning "Directory $DeploymentDir already exists. Cleaning up..."
        Remove-Item -Path "$DeploymentDir\*" -Recurse -Force -ErrorAction SilentlyContinue
    }
    else {
        New-Item -ItemType Directory -Path $DeploymentDir | Out-Null
    }
    
    Set-Location $DeploymentDir
    Write-Success "Created and entered directory: $DeploymentDir"

    # Create docker-compose.yml
    Write-Info "Creating docker-compose.yml file..."
    $dockerComposeContent = @"
version: '3.8'

services:
  # FastAPI Backend Service
  backend:
    image: $BackendImage
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=sqlite:///mlflow.db
    volumes:
      - mlruns:/app/mlruns
      - models:/app/models
    networks:
      - app-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  # Streamlit Frontend Service  
  frontend:
    image: $FrontendImage
    ports:
      - "8501:8501"
    environment:
      - BACKEND_URL=http://backend:8000
    depends_on:
      - backend
    networks:
      - app-network
    restart: unless-stopped

networks:
  app-network:
    driver: bridge

volumes:
  mlruns:
  models:
"@

    $dockerComposeContent | Out-File -FilePath "docker-compose.yml" -Encoding utf8
    Write-Success "Docker Compose file created"

    # Pull images
    Write-Info "Pulling Docker images..."
    Write-Host "📥 Pulling backend image: $BackendImage" -ForegroundColor $White
    docker pull $BackendImage
    
    Write-Host "📥 Pulling frontend image: $FrontendImage" -ForegroundColor $White
    docker pull $FrontendImage
    
    Write-Success "Both images pulled successfully"

    # Stop any existing containers
    Write-Info "Stopping any existing Dig-Reg containers..."
    try {
        if ($ComposeCmd -eq "docker-compose") {
            docker-compose down --remove-orphans 2>$null
        } else {
            docker compose down --remove-orphans 2>$null
        }
    }
    catch {
        # Ignore errors if no containers exist
    }

    # Start services
    Write-Info "Starting Dig-Reg services..."
    if ($ComposeCmd -eq "docker-compose") {
        docker-compose up -d
    } else {
        docker compose up -d
    }

    # Wait for services to start
    Write-Info "Waiting for services to start..."
    Start-Sleep -Seconds 5

    # Check service status
    Write-Info "Checking service status..."
    if ($ComposeCmd -eq "docker-compose") {
        docker-compose ps
    } else {
        docker compose ps
    }

    # Health checks
    Write-Info "Performing health checks..."
    
    # Backend health check
    Write-Host "🔍 Backend health check... " -NoNewline -ForegroundColor $White
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -TimeoutSec 10 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Success "✅ Backend is healthy"
        } else {
            Write-Warning "⚠️ Backend health check returned status: $($response.StatusCode)"
        }
    }
    catch {
        Write-Warning "⚠️ Backend health check failed (may still be starting)"
    }

    # Frontend health check
    Write-Host "🔍 Frontend availability check... " -NoNewline -ForegroundColor $White
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 10 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Success "✅ Frontend is accessible"
        } else {
            Write-Warning "⚠️ Frontend returned status: $($response.StatusCode)"
        }
    }
    catch {
        Write-Warning "⚠️ Frontend not yet accessible (may still be starting)"
    }

    # Success message
    Write-Host ""
    Write-Host "🎉 Deployment complete!" -ForegroundColor $Green
    Write-Host ""
    Write-Host "📱 Access the application:" -ForegroundColor $Blue
    Write-Host "   Frontend (Streamlit): http://localhost:8501" -ForegroundColor $White
    Write-Host "   Backend API:          http://localhost:8000" -ForegroundColor $White
    Write-Host "   API Documentation:    http://localhost:8000/docs" -ForegroundColor $White
    Write-Host ""
    Write-Host "🔧 Management commands:" -ForegroundColor $Blue
    if ($ComposeCmd -eq "docker-compose") {
        Write-Host "   View logs:      docker-compose logs -f" -ForegroundColor $White
        Write-Host "   Stop services:  docker-compose down" -ForegroundColor $White
        Write-Host "   Restart:        docker-compose restart" -ForegroundColor $White
        Write-Host "   Status:         docker-compose ps" -ForegroundColor $White
    } else {
        Write-Host "   View logs:      docker compose logs -f" -ForegroundColor $White
        Write-Host "   Stop services:  docker compose down" -ForegroundColor $White
        Write-Host "   Restart:        docker compose restart" -ForegroundColor $White
        Write-Host "   Status:         docker compose ps" -ForegroundColor $White
    }
    Write-Host ""
    Write-Success "Dig-Reg is now running! 🚀"

    # Ask if user wants to open browser
    $openBrowser = Read-Host "Would you like to open the application in your default browser? (y/N)"
    if ($openBrowser -eq 'y' -or $openBrowser -eq 'Y') {
        Write-Info "Opening browser..."
        Start-Process "http://localhost:8501"
    }

}
catch {
    Write-Error "Deployment failed: $($_.Exception.Message)"
    Write-Host "Please check the error above and try again." -ForegroundColor $Yellow
    Write-Host "For help, visit: https://github.com/munikumar229/Dig-Reg/blob/main/docs/WINDOWS_DEPLOYMENT.md" -ForegroundColor $Yellow
    exit 1
}
finally {
    # Return to original directory
    Set-Location ..
}