from gnn_api_simple import app

# This is the entry point for Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 