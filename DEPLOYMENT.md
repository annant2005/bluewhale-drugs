# Deployment Guide

This guide will help you deploy your Drug Toxicity Prediction application with the backend on Vercel and frontend on Streamlit Cloud.

## Prerequisites

1. **GitHub Account** - Both Vercel and Streamlit Cloud deploy from GitHub repositories
2. **Vercel Account** - Sign up at [vercel.com](https://vercel.com)
3. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)

## Step 1: Prepare Your Repository

1. Push your code to a GitHub repository
2. Make sure your repository structure looks like this:
   ```
   your-repo/
   ├── backend/
   │   ├── gnn_api_simple.py
   │   ├── requirements.txt
   │   ├── vercel.json
   │   └── api/
   │       └── index.py
   ├── frontend/
   │   ├── app_advanced.py
   │   ├── app.py
   │   ├── requirements.txt
   │   └── .streamlit/
   │       └── config.toml
   └── README.md
   ```

## Step 2: Deploy Backend to Vercel

### Option A: Deploy via Vercel Dashboard

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository
4. Configure the project:
   - **Framework Preset**: Other
   - **Root Directory**: `backend`
   - **Build Command**: Leave empty (Vercel will auto-detect)
   - **Output Directory**: Leave empty
   - **Install Command**: `pip install -r requirements.txt`
5. Click "Deploy"

### Option B: Deploy via Vercel CLI

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Deploy:
   ```bash
   vercel
   ```

4. Follow the prompts to link to your GitHub repository

### Get Your Backend URL

After deployment, Vercel will provide you with a URL like:
`https://your-app-name.vercel.app`

## Step 3: Deploy Frontend to Streamlit Cloud

### Option A: Deploy via Streamlit Cloud Dashboard

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
2. Click "New app"
3. Configure the deployment:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main` (or your default branch)
   - **Main file path**: `frontend/app_advanced.py`
   - **App URL**: Choose a custom subdomain (optional)
4. Click "Deploy"

### Option B: Deploy via Streamlit CLI

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```

2. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

3. Deploy:
   ```bash
   streamlit deploy app_advanced.py
   ```

## Step 4: Configure Environment Variables

### For Streamlit Cloud (Frontend)

1. In your Streamlit Cloud dashboard, go to your app settings
2. Add the following environment variable:
   - **Key**: `BACKEND_URL`
   - **Value**: Your Vercel backend URL (e.g., `https://your-app-name.vercel.app`)

### For Vercel (Backend) - Optional

If you need to configure any backend-specific environment variables:
1. In your Vercel dashboard, go to your project settings
2. Navigate to "Environment Variables"
3. Add any required variables

## Step 5: Test Your Deployment

1. **Test Backend**: Visit your Vercel URL + `/docs` to see the FastAPI documentation
2. **Test Frontend**: Visit your Streamlit Cloud URL to test the complete application

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are in the respective `requirements.txt` files
2. **Model Loading**: Ensure your trained models are included in the repository
3. **CORS Issues**: The backend is configured to allow all origins
4. **Timeout Issues**: Vercel has a 30-second timeout limit for serverless functions

### Debugging

1. **Check Vercel Logs**: In your Vercel dashboard, go to "Functions" to see deployment logs
2. **Check Streamlit Logs**: In your Streamlit Cloud dashboard, check the "Logs" tab
3. **Local Testing**: Test locally first using the provided scripts

## Alternative Deployment Options

### Backend Alternatives
- **Railway**: Good for Python apps with longer execution times
- **Render**: Free tier available, good for ML applications
- **Heroku**: Paid service, but very reliable

### Frontend Alternatives
- **Hugging Face Spaces**: Good for ML applications
- **Gradio**: Alternative to Streamlit for ML demos
- **Heroku**: Can host Streamlit apps

## Cost Considerations

- **Vercel**: Free tier includes 100GB bandwidth and 100 serverless function executions per day
- **Streamlit Cloud**: Free tier available with some limitations
- **Model Storage**: Consider using cloud storage (AWS S3, Google Cloud Storage) for large model files

## Security Notes

1. **API Keys**: Never commit API keys to your repository
2. **Environment Variables**: Use environment variables for sensitive configuration
3. **CORS**: Configure CORS properly for production
4. **Rate Limiting**: Consider implementing rate limiting for your API

## Monitoring and Maintenance

1. **Health Checks**: Implement health check endpoints
2. **Logging**: Add proper logging to your applications
3. **Monitoring**: Set up alerts for downtime
4. **Updates**: Regularly update dependencies for security patches 