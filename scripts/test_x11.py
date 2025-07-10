#!/usr/bin/env python3
"""
Quick test to verify X11/VcXsrv connection and then test teleoperation.
"""

import subprocess
import sys
import os

def test_x11_connection():
    """Test if X11 connection is working."""
    print("🧪 Testing X11 connection...")
    
    try:
        # Try to run a simple X11 command
        result = subprocess.run(['xset', 'q'], capture_output=True, timeout=5)
        if result.returncode == 0:
            print("✅ X11 connection working!")
            return True
        else:
            print("❌ X11 connection failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ X11 connection timeout/not found")
        return False

def test_teleoperation():
    """Test teleoperation with visual rendering."""
    print("\n🎮 Testing teleoperation with visuals...")
    
    # Clear problematic environment variables
    env = os.environ.copy()
    env.pop('MUJOCO_GL', None)
    env.pop('PYOPENGL_PLATFORM', None)
    env.pop('LIBGL_ALWAYS_SOFTWARE', None)
    
    try:
        # Try to run teleoperation example
        cmd = ['python', 'teleoperation_example.py']
        proc = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Send input: choose environment 1, don't use headless mode
        stdout, stderr = proc.communicate(input="1\nn\n", timeout=10)
        
        if "Environment started" in stdout:
            print("✅ Visual teleoperation working!")
            return True
        else:
            print("❌ Visual teleoperation failed")
            print("Stdout:", stdout[-200:])  # Last 200 chars
            print("Stderr:", stderr[-200:])  # Last 200 chars
            return False
            
    except subprocess.TimeoutExpired:
        proc.kill()
        print("❌ Teleoperation test timed out")
        return False
    except Exception as e:
        print(f"❌ Teleoperation test error: {e}")
        return False

def main():
    print("🔧 VcXsrv + OGBench Teleoperation Test")
    print("=" * 50)
    
    # Check DISPLAY variable
    display = os.environ.get('DISPLAY')
    if display:
        print(f"📺 DISPLAY is set to: {display}")
    else:
        print("❌ DISPLAY variable not set")
        return 1
    
    # Test X11 connection
    if not test_x11_connection():
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure VcXsrv is running on Windows")
        print("2. Check 'Disable access control' is enabled")
        print("3. Allow VcXsrv through Windows Firewall")
        print("4. Try restarting VcXsrv")
        return 1
    
    # Test teleoperation
    if test_teleoperation():
        print("\n🎉 SUCCESS! Visual teleoperation is working!")
        print("You can now run the full teleoperation scripts with visuals:")
        print("  python teleoperation_example.py")
        print("  python intervention_example.py")
        return 0
    else:
        print("\n😔 Visual rendering still not working.")
        print("You can still use headless mode:")
        print("  python headless_test.py")
        return 1

if __name__ == "__main__":
    exit(main()) 