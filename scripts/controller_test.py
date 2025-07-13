#!/usr/bin/env python3
"""
Controller detection and testing script.
This will help diagnose Xbox 360 controller issues.
"""

import pygame
import time

def test_controller_detection():
    """Test if pygame can detect controllers."""
    print("=== Controller Detection Test ===")
    
    pygame.init()
    pygame.joystick.init()
    
    # Check for controllers
    joystick_count = pygame.joystick.get_count()
    print(f"🎮 Controllers detected: {joystick_count}")
    
    if joystick_count == 0:
        print("❌ No controllers found!")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure your Xbox 360 controller is plugged in")
        print("2. Try: sudo apt install xboxdrv")
        print("3. Check: lsusb | grep -i xbox")
        print("4. Test with: jstest /dev/input/js0")
        return False
    
    # Test each controller
    for i in range(joystick_count):
        try:
            controller = pygame.joystick.Joystick(i)
            controller.init()
            
            print(f"\n✅ Controller {i}:")
            print(f"   Name: {controller.get_name()}")
            print(f"   Axes: {controller.get_numaxes()}")
            print(f"   Buttons: {controller.get_numbuttons()}")
            print(f"   Hats: {controller.get_numhats()}")
            
            # Test if it's an Xbox controller
            name_lower = controller.get_name().lower()
            if 'xbox' in name_lower or '360' in name_lower:
                print("   🎯 Xbox 360 controller detected!")
            
        except Exception as e:
            print(f"❌ Error initializing controller {i}: {e}")
    
    return True

def test_controller_input():
    """Test controller input in real-time."""
    print("\n=== Controller Input Test ===")
    
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("❌ No controllers to test")
        return
    
    # Use first controller
    controller = pygame.joystick.Joystick(0)
    controller.init()
    
    print(f"🎮 Testing: {controller.get_name()}")
    print("Move the left stick and press buttons...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            pygame.event.pump()
            
            # Read left stick
            if controller.get_numaxes() >= 2:
                x_axis = controller.get_axis(0)
                y_axis = controller.get_axis(1)
                
                if abs(x_axis) > 0.1 or abs(y_axis) > 0.1:
                    print(f"\r🕹️  Left Stick: X={x_axis:+.2f}, Y={y_axis:+.2f}    ", end="", flush=True)
            
            # Read buttons
            if controller.get_numbuttons() > 0:
                for i in range(min(controller.get_numbuttons(), 8)):  # Check first 8 buttons
                    if controller.get_button(i):
                        print(f"\n🔘 Button {i} pressed!")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n✅ Controller test completed")

def main():
    print("=== Xbox 360 Controller Diagnostic ===")
    
    # Test detection
    detected = test_controller_detection()
    
    if detected:
        try_input = input("\nTest controller input? (y/n): ").lower().strip()
        if try_input == 'y':
            test_controller_input()
    
    print("\n=== System Commands to Try ===")
    print("# Check USB devices:")
    print("lsusb | grep -i xbox")
    print("\n# Check input devices:")
    print("ls /dev/input/js*")
    print("\n# Test with jstest (if available):")
    print("sudo apt install joystick")
    print("jstest /dev/input/js0")

if __name__ == "__main__":
    main() 