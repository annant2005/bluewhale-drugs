import subprocess
import time
import sys
import os

def start_simple_api():
    """Start the simple toxicity API"""
    print("🚀 Starting Simple Toxicity API...")
    api_process = subprocess.Popen([
        sys.executable, "backend/gnn_api_simple.py"
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
    print("💊 Starting Simple Drug Toxicity Prediction System")
    print("=" * 60)
    
    # Start API
    api_process = start_simple_api()
    
    # Start frontend
    frontend_process = start_frontend()
    
    print("\n✅ System is starting up!")
    print("📊 Frontend will be available at: http://localhost:8501")
    print("🔌 API will be available at: http://localhost:8000")
    print("\n🎯 Test compounds:")
    print("   - Nicotine: CN1CCCC1C2=CN=CC=C2")
    print("   - Benzene: C1=CC=CC=C1")
    print("   - Ethanol: CCO")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        api_process.terminate()
        frontend_process.terminate()
        print("✅ System stopped.") 