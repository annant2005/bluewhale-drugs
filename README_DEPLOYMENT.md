# Quick Deployment Guide

## 🚀 Deploy Your Drug Toxicity Prediction App

This guide will help you deploy your application with:
- **Backend**: Vercel (FastAPI)
- **Frontend**: Streamlit Cloud

## Prerequisites

1. **GitHub Account** - Push your code to GitHub first
2. **Vercel Account** - [Sign up here](https://vercel.com)
3. **Streamlit Cloud Account** - [Sign up here](https://share.streamlit.io)

## Quick Start

### Option 1: Use Deployment Scripts

**Windows:**
```bash
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### Option 2: Manual Deployment

#### Step 1: Deploy Backend to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. Set **Root Directory** to `backend`
5. Click "Deploy"
6. Note your Vercel URL (e.g., `https://your-app.vercel.app`)

#### Step 2: Deploy Frontend to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository
4. Set **Main file path** to `frontend/app_advanced.py`
5. Click "Deploy"

#### Step 3: Configure Environment Variables

In your Streamlit Cloud dashboard:
1. Go to your app settings
2. Add environment variable:
   - **Key**: `BACKEND_URL`
   - **Value**: Your Vercel URL from Step 1

## Testing

1. **Backend**: Visit `your-vercel-url/docs` for API documentation
2. **Frontend**: Visit your Streamlit Cloud URL to test the app

## Troubleshooting

- **Import Errors**: Check `requirements.txt` files
- **Model Loading**: Ensure trained models are in the repository
- **CORS Issues**: Backend is configured to allow all origins
- **Timeout**: Vercel has 30-second limits for serverless functions

## File Structure

```
your-repo/
├── backend/
│   ├── gnn_api_simple.py      # FastAPI app
│   ├── requirements.txt       # Backend dependencies
│   ├── vercel.json           # Vercel config
│   └── api/index.py          # Vercel entry point
├── frontend/
│   ├── app_advanced.py       # Main Streamlit app
│   ├── requirements.txt      # Frontend dependencies
│   └── .streamlit/config.toml # Streamlit config
├── deploy.sh                 # Linux/Mac deployment script
├── deploy.bat               # Windows deployment script
└── DEPLOYMENT.md            # Detailed deployment guide
```

## Support

- 📖 [Detailed Deployment Guide](DEPLOYMENT.md)
- 🔗 [Vercel Documentation](https://vercel.com/docs)
- 🔗 [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud) 