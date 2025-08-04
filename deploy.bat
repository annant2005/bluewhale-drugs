@echo off
chcp 65001 >nul
echo 🚀 Drug Toxicity Prediction App Deployment Script
echo ==================================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Git is not installed. Please install Git first.
    pause
    exit /b 1
)

REM Check if we're in a git repository
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
    echo ❌ Not in a git repository. Please initialize git and push to GitHub first.
    echo Run these commands:
    echo   git init
    echo   git add .
    echo   git commit -m "Initial commit"
    echo   git remote add origin ^<your-github-repo-url^>
    echo   git push -u origin main
    pause
    exit /b 1
)

echo ✅ Git repository is ready
echo.

echo Choose deployment option:
echo 1. Deploy Backend to Vercel
echo 2. Deploy Frontend to Streamlit Cloud
echo 3. Deploy Both (Backend + Frontend)
echo 4. Configure Environment Variables
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto deploy_backend
if "%choice%"=="2" goto deploy_frontend
if "%choice%"=="3" goto deploy_both
if "%choice%"=="4" goto configure_env
if "%choice%"=="5" goto exit
echo ❌ Invalid choice. Please run the script again.
pause
exit /b 1

:deploy_backend
echo.
echo 🔧 Deploying Backend to Vercel...
echo ==================================
echo.

REM Check if Vercel CLI is installed
vercel --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Vercel CLI is not installed. Installing...
    npm install -g vercel
)

cd backend

if not exist "vercel.json" (
    echo ❌ vercel.json not found in backend directory
    pause
    exit /b 1
)

echo 📦 Deploying to Vercel...
vercel --prod

cd ..

echo ✅ Backend deployment initiated!
echo 📋 Please note your Vercel URL for the next step
goto end

:deploy_frontend
echo.
echo 🎨 Deploying Frontend to Streamlit Cloud...
echo ============================================
echo.

REM Check if Streamlit is installed
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit is not installed. Installing...
    pip install streamlit
)

cd frontend

if not exist "app_advanced.py" (
    echo ❌ app_advanced.py not found in frontend directory
    pause
    exit /b 1
)

echo 📦 Deploying to Streamlit Cloud...
streamlit deploy app_advanced.py

cd ..

echo ✅ Frontend deployment initiated!
goto end

:deploy_both
call :deploy_backend
call :deploy_frontend
call :configure_env
goto end

:configure_env
echo.
echo ⚙️  Environment Configuration
echo =============================
echo.

set /p BACKEND_URL="Enter your Vercel backend URL (e.g., https://your-app.vercel.app): "

if "%BACKEND_URL%"=="" (
    echo ❌ Backend URL is required
    pause
    exit /b 1
)

echo.
echo 📋 Environment Variables to Configure:
echo ======================================
echo.
echo For Streamlit Cloud (Frontend):
echo   Key: BACKEND_URL
echo   Value: %BACKEND_URL%
echo.
echo Instructions:
echo 1. Go to your Streamlit Cloud dashboard
echo 2. Select your app
echo 3. Go to Settings ^> Secrets
echo 4. Add the environment variable above
echo.
echo For Vercel (Backend) - Optional:
echo 1. Go to your Vercel dashboard
echo 2. Select your project
echo 3. Go to Settings ^> Environment Variables
echo 4. Add any required variables
goto end

:end
echo.
echo 🎉 Deployment process completed!
echo.
echo 📚 Next Steps:
echo 1. Wait for deployments to complete
echo 2. Configure environment variables as shown above
echo 3. Test your application
echo 4. Check the DEPLOYMENT.md file for troubleshooting
echo.
echo 🔗 Useful Links:
echo - Vercel Dashboard: https://vercel.com/dashboard
echo - Streamlit Cloud: https://share.streamlit.io
echo - Deployment Guide: DEPLOYMENT.md
pause
exit /b 0

:exit
echo 👋 Goodbye!
pause
exit /b 0 