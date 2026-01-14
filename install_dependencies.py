import subprocess
import sys

# List of libraries used in your Thales app
libraries = [
    "streamlit",
    "yfinance",
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn"
]

def install(package):
    print(f"📦 Installing {package}...")
    try:
        # This command is equivalent to running 'pip install package' in the terminal
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully!")
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}.")

if __name__ == "__main__":
    print("--- Starting Auto-Installation ---")
    
    # Update pip first (optional but recommended)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    except:
        pass

    # Install each library
    for lib in libraries:
        install(lib)
        
    print("\n🎉 All libraries installed! You can now run: streamlit run app.py")
