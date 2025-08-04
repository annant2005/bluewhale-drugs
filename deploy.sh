#!/bin/bash

# Drug Toxicity Prediction App Deployment Script
# This script helps deploy the backend to Vercel and frontend to Streamlit Cloud

echo "🚀 Drug Toxicity Prediction App Deployment Script"
echo "=================================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository. Please initialize git and push to GitHub first."
    echo "Run these commands:"
    echo "  git init"
    echo "  git add ."
    echo "  git commit -m 'Initial commit'"
    echo "  git remote add origin <your-github-repo-url>"
    echo "  git push -u origin main"
    exit 1
fi

# Check if we have uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  You have uncommitted changes. Please commit them first:"
    echo "  git add ."
    echo "  git commit -m 'Update before deployment'"
    echo "  git push"
    exit 1
fi

echo "✅ Git repository is ready"

# Function to deploy backend to Vercel
deploy_backend() {
    echo ""
    echo "🔧 Deploying Backend to Vercel..."
    echo "=================================="
    
    if ! command -v vercel &> /dev/null; then
        echo "❌ Vercel CLI is not installed. Installing..."
        npm install -g vercel
    fi
    
    cd backend
    
    if [ ! -f "vercel.json" ]; then
        echo "❌ vercel.json not found in backend directory"
        exit 1
    fi
    
    echo "📦 Deploying to Vercel..."
    vercel --prod
    
    cd ..
    
    echo "✅ Backend deployment initiated!"
    echo "📋 Please note your Vercel URL for the next step"
}

# Function to deploy frontend to Streamlit Cloud
deploy_frontend() {
    echo ""
    echo "🎨 Deploying Frontend to Streamlit Cloud..."
    echo "============================================"
    
    if ! command -v streamlit &> /dev/null; then
        echo "❌ Streamlit is not installed. Installing..."
        pip install streamlit
    fi
    
    cd frontend
    
    if [ ! -f "app_advanced.py" ]; then
        echo "❌ app_advanced.py not found in frontend directory"
        exit 1
    fi
    
    echo "📦 Deploying to Streamlit Cloud..."
    streamlit deploy app_advanced.py
    
    cd ..
    
    echo "✅ Frontend deployment initiated!"
}

# Function to configure environment variables
configure_env() {
    echo ""
    echo "⚙️  Environment Configuration"
    echo "============================="
    
    read -p "Enter your Vercel backend URL (e.g., https://your-app.vercel.app): " BACKEND_URL
    
    if [ -z "$BACKEND_URL" ]; then
        echo "❌ Backend URL is required"
        exit 1
    fi
    
    echo ""
    echo "📋 Environment Variables to Configure:"
    echo "======================================"
    echo ""
    echo "For Streamlit Cloud (Frontend):"
    echo "  Key: BACKEND_URL"
    echo "  Value: $BACKEND_URL"
    echo ""
    echo "Instructions:"
    echo "1. Go to your Streamlit Cloud dashboard"
    echo "2. Select your app"
    echo "3. Go to Settings > Secrets"
    echo "4. Add the environment variable above"
    echo ""
    echo "For Vercel (Backend) - Optional:"
    echo "1. Go to your Vercel dashboard"
    echo "2. Select your project"
    echo "3. Go to Settings > Environment Variables"
    echo "4. Add any required variables"
}

# Main menu
echo ""
echo "Choose deployment option:"
echo "1. Deploy Backend to Vercel"
echo "2. Deploy Frontend to Streamlit Cloud"
echo "3. Deploy Both (Backend + Frontend)"
echo "4. Configure Environment Variables"
echo "5. Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        deploy_backend
        ;;
    2)
        deploy_frontend
        ;;
    3)
        deploy_backend
        deploy_frontend
        configure_env
        ;;
    4)
        configure_env
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment process completed!"
echo ""
echo "📚 Next Steps:"
echo "1. Wait for deployments to complete"
echo "2. Configure environment variables as shown above"
echo "3. Test your application"
echo "4. Check the DEPLOYMENT.md file for troubleshooting"
echo ""
echo "🔗 Useful Links:"
echo "- Vercel Dashboard: https://vercel.com/dashboard"
echo "- Streamlit Cloud: https://share.streamlit.io"
echo "- Deployment Guide: DEPLOYMENT.md" 