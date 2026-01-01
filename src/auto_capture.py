#!/usr/bin/env python3
"""
Automate ScummVM room captures using subprocess and screenshots.
Requires: pip install pyautogui pillow
"""
import subprocess
import time
import os

try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False
    print("Install pyautogui for automation: pip install pyautogui")

ROOMS = [100, 105, 110, 200, 300, 400, 500, 600, 700, 800, 900]

def capture_with_screencapture(room_num, output_dir):
    """Use macOS screencapture to grab the screen."""
    filepath = os.path.join(output_dir, f"LSL5-{room_num:04d}.png")
    subprocess.run(["screencapture", "-x", "-R", "0,0,640,400", filepath])
    return filepath

def main():
    output_dir = "extracts/screenshots"
    os.makedirs(output_dir, exist_ok=True)

    print("Manual capture instructions:")
    print("1. Launch ScummVM with Larry 5")
    print("2. Open debugger with Ctrl+D")
    print("3. For each room, run these commands:")
    print()

    for room in ROOMS:
        print(f"   room {room}")
        print(f"   go")
        print(f"   # Then press Alt+S to save screenshot")
        print()

if __name__ == "__main__":
    main()
