# deploy_system.py
import subprocess
import sys
import time
import webbrowser
from threading import Thread

def run_command(command, description):
    """Run a system command and handle output"""
    print(f"🚀 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def start_backend_services():
    """Start all backend services"""
    services = []
    
    # Start real-time simulator
    realtime_process = subprocess.Popen([sys.executable, "realtime_simulator.py"])
    services.append(realtime_process)
    
    # Start advanced models (if needed)
    models_process = subprocess.Popen([sys.executable, "advanced_models.py"])
    services.append(models_process)
    
    return services

def open_browser_dashboards():
    """Open all dashboard interfaces in browser"""
    time.sleep(3)  # Wait for services to start
    
    dashboards = [
        "http://localhost:8000/dashboard.html",
        "http://localhost:8000/cesium_map/index.html"
    ]
    
    for dashboard in dashboards:
        webbrowser.open(dashboard)
        time.sleep(1)

def main():
    print("🎯 DEPLOYING COMPLETE AI URBAN HAZARD SYSTEM")
    print("=" * 60)
    
    # Step 1: Run data generation and AI training
    if not run_command("python run.py", "Data generation and AI training"):
        print("❌ Failed to initialize system")
        return
    
    # Step 2: Train advanced models
    if not run_command("python advanced_models.py", "Advanced AI model training"):
        print("⚠️ Advanced models training failed, continuing with basic models")
    
    # Step 3: Test emergency system
    if not run_command("python emergency_response.py", "Emergency system testing"):
        print("⚠️ Emergency system test failed, but continuing")
    
    # Step 4: Start backend services
    print("🔧 Starting backend services...")
    services = start_backend_services()
    
    # Step 5: Start web server
    print("🌐 Starting web server...")
    server_process = subprocess.Popen([sys.executable, "-m", "http.server", "8000"])
    
    # Step 6: Open dashboards
    print("📱 Opening dashboard interfaces...")
    Thread(target=open_browser_dashboards).start()
    
    print("\n🎉 SYSTEM DEPLOYMENT COMPLETE!")
    print("=" * 60)
    print("📊 Main Dashboard: http://localhost:8000/dashboard.html")
    print("🌍 3D Digital Twin: http://localhost:8000/cesium_map/index.html")
    print("🔗 Real-time API: ws://localhost:8765")
    print("\nPress Ctrl+C to stop all services")
    
    try:
        # Keep services running
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        for service in services:
            service.terminate()
        server_process.terminate()

if __name__ == "__main__":
    main()