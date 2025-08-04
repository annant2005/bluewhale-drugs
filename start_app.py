import subprocess
import time
import sys
import os

def start_api():
    """Start the comprehensive API"""
    print("🚀 Starting Comprehensive Toxicity API...")
    api_process = subprocess.Popen([
        sys.executable, "backend/gnn_api_comprehensive.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for the API to start
    time.sleep(3)
    return api_process

def start_frontend():
    """Start the Streamlit frontend"""
    print("🌐 Starting Streamlit Frontend...")
    frontend_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "frontend/app_advanced.py", 
        "--server.port", "8501", "--server.address", "localhost"
    ])
    return frontend_process

if __name__ == "__main__":
    print("💊 Starting Comprehensive Drug Toxicity Prediction System")
    print("=" * 60)
    
    # Start API
    api_process = start_api()
    
    try:
        # Start frontend
        frontend_process = start_frontend()
        
        print("\n✅ System is starting up!")
        print("📊 API: http://localhost:8000")
        print("🌐 Frontend: http://localhost:8501")
        print("\n⏳ Waiting for services to start...")
        print("Press Ctrl+C to stop all services")
        
        # Keep running
        try:
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down services...")
            frontend_process.terminate()
            api_process.terminate()
            print("✅ Services stopped")
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        api_process.terminate() 