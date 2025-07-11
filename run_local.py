#!/usr/bin/env python3
"""
Local development script for Trustworthy AI Blockchain-Integrated Federated Learning
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        "streamlit>=1.28.0",
        "numpy>=1.24.0", 
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "networkx>=3.1.0"
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    return True

def run_streamlit():
    """Run the Streamlit application"""
    print("\nStarting Streamlit application...")
    print("Access the application at: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running Streamlit: {e}")

def main():
    """Main function"""
    print("=" * 60)
    print("Trustworthy AI: Blockchain-Integrated Federated Learning")
    print("=" * 60)
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("Error: app.py not found in current directory")
        print("Please run this script from the project root directory")
        return
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually.")
        return
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()