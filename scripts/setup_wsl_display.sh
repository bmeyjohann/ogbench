#!/bin/bash
# Setup script for fixing display issues in WSL
# This helps resolve OpenGL/MuJoCo rendering problems

echo "🔧 OGBench WSL Display Setup Helper"
echo "=================================="

# Check if we're in WSL
if grep -qi microsoft /proc/version; then
    echo "✅ WSL detected"
else
    echo "❌ Not running in WSL - this script is for WSL environments"
    exit 1
fi

echo ""
echo "This script helps fix OpenGL/MuJoCo rendering issues in WSL."
echo "You have several options:"
echo ""
echo "1. Install and configure X11 forwarding (recommended)"
echo "2. Use headless mode (no visual rendering)"
echo "3. Use VcXsrv/Xming for Windows X server"
echo ""

read -p "Choose option (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "🔄 Setting up X11 forwarding..."
        
        # Check if X11 forwarding is available
        if [ -z "$DISPLAY" ]; then
            echo "Setting DISPLAY variable..."
            export DISPLAY=:0
            echo "export DISPLAY=:0" >> ~/.bashrc
        fi
        
        # Install X11 utilities if not present
        echo "Installing X11 utilities..."
        sudo apt update
        sudo apt install -y x11-apps mesa-utils
        
        echo ""
        echo "✅ X11 setup complete!"
        echo "⚠️  You may still need to:"
        echo "   1. Install VcXsrv or Xming on Windows"
        echo "   2. Configure Windows firewall to allow X11"
        echo "   3. Restart your terminal"
        
        # Test X11
        echo ""
        echo "Testing X11 connection..."
        if command -v xeyes >/dev/null 2>&1; then
            echo "Run 'xeyes' to test if X11 forwarding works"
        fi
        ;;
        
    2)
        echo ""
        echo "🖥️  Configuring headless mode..."
        echo "Setting environment variables for headless operation..."
        
        # Set environment variables for headless MuJoCo
        export MUJOCO_GL=egl
        export PYOPENGL_PLATFORM=egl
        
        echo "export MUJOCO_GL=egl" >> ~/.bashrc
        echo "export PYOPENGL_PLATFORM=egl" >> ~/.bashrc
        
        echo ""
        echo "✅ Headless mode configured!"
        echo "The teleoperation system will work without visual rendering."
        echo "Restart your terminal and run: python headless_test.py"
        ;;
        
    3)
        echo ""
        echo "🪟 VcXsrv/Xming setup instructions:"
        echo ""
        echo "1. Download and install VcXsrv from:"
        echo "   https://sourceforge.net/projects/vcxsrv/"
        echo ""
        echo "2. Start VcXsrv with these settings:"
        echo "   - Display number: 0"
        echo "   - Start no client: checked"
        echo "   - Disable access control: checked"
        echo ""
        echo "3. Set DISPLAY variable:"
        export DISPLAY=$(ip route | grep default | awk '{print $3}'):0
        echo "export DISPLAY=$(ip route | grep default | awk '{print $3}'):0" >> ~/.bashrc
        echo "   DISPLAY set to: $DISPLAY"
        echo ""
        echo "4. Test with: xeyes"
        ;;
        
    *)
        echo "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🧪 Testing the setup..."
cd "$(dirname "$0")"

# Test if ogbench is installed
if python -c "import ogbench" 2>/dev/null; then
    echo "✅ OGBench is installed"
    
    # Run headless test
    echo "Running headless test..."
    python headless_test.py
else
    echo "❌ OGBench not found. Please install it first:"
    echo "   cd ogbench && pip install -e ."
fi

echo ""
echo "🎯 Setup complete! Try running the teleoperation scripts:"
echo "   python teleoperation_example.py"
echo "   python intervention_example.py"
echo "   python headless_test.py" 