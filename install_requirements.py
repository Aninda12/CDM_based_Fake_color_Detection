import sys
import platform
import subprocess

def install_requirements():
    """Install common dependencies from requirements.txt"""
    try:
        print("Installing common dependencies...")
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    except subprocess.CalledProcessError:
        print("Error installing base dependencies.")
        sys.exit(1)

def install_mac_specific():
    """Install optional Mac-specific dependencies if applicable"""
    if sys.platform == "darwin" and platform.machine() == "arm64":
        print("Detected macOS on Apple Silicon (ARM64). Installing mac-specific dependencies...")
        try:
            subprocess.run(["pip", "install", "tensorflow-macos>=2.9.0", "tensorflow-metal>=0.5.0"], check=True)
        except subprocess.CalledProcessError:
            print("Error installing Mac-specific dependencies.")
            sys.exit(1)
    else:
        print("Installing standard TensorFlow package...")
        try:
            subprocess.run(["pip", "install", "tensorflow>=2.9.0"], check=True)
        except subprocess.CalledProcessError:
            print("Error installing TensorFlow.")
            sys.exit(1)

if __name__ == "__main__":
    install_requirements()
    install_mac_specific()
    print("Installation complete.")
