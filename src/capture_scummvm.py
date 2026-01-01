#!/usr/bin/env python3
"""
Automate ScummVM room captures on macOS using osascript.

Usage:
1. Start ScummVM and launch Larry 5
2. Run this script: python src/capture_scummvm.py
"""
import subprocess
import time
import os

# KQ5 room numbers (0-99 for most rooms)
ROOMS = list(range(1, 100))


def run_applescript(script):
    """Run AppleScript via osascript."""
    subprocess.run(["osascript", "-e", script], check=True)


def keystroke(text):
    """Send keystrokes to ScummVM."""
    script = f'''
    tell application "System Events"
        set frontmost of process "scummvm" to true
        delay 0.1
        tell process "scummvm"
            keystroke "{text}"
        end tell
    end tell
    '''
    run_applescript(script)


def key_code(code, modifier=None):
    """Send a key code to ScummVM."""
    if modifier:
        script = f'''
        tell application "System Events"
            set frontmost of process "scummvm" to true
            delay 0.1
            tell process "scummvm"
                key code {code} using {modifier} down
            end tell
        end tell
        '''
    else:
        script = f'''
        tell application "System Events"
            set frontmost of process "scummvm" to true
            delay 0.1
            tell process "scummvm"
                key code {code}
            end tell
        end tell
        '''
    run_applescript(script)


def press_enter():
    key_code(36)


def press_ctrl_d():
    key_code(2, "control")


def press_alt_s():
    key_code(1, "option")


def activate_scummvm():
    """Activate ScummVM and ensure it has focus."""
    run_applescript('''
    tell application "System Events"
        set frontmost of process "scummvm" to true
    end tell
    ''')
    time.sleep(0.3)


def capture_room(room_num):
    """Capture a single room."""
    print(f"Capturing room {room_num}...")

    # Type room command
    keystroke(f"room {room_num}")
    time.sleep(0.1)
    press_enter()
    time.sleep(0.2)

    # Type go command
    keystroke("go")
    time.sleep(0.1)
    press_enter()
    time.sleep(1.5)  # Wait for room to render

    # Take screenshot
    press_alt_s()
    time.sleep(0.3)

    # Reopen debugger
    press_ctrl_d()
    time.sleep(0.3)


def main():
    print("ScummVM Room Capture Script")
    print("===========================")
    print()
    print("Prerequisites:")
    print("1. ScummVM must be running with KQ5 loaded")
    print("2. Grant Terminal accessibility permissions if prompted")
    print()
    print("Starting in 3 seconds...")
    time.sleep(3)

    # Activate ScummVM
    activate_scummvm()
    time.sleep(0.5)

    # Open debugger
    print("Opening debugger...")
    press_ctrl_d()
    time.sleep(0.5)

    # Capture each room
    for i, room in enumerate(ROOMS):
        print(f"[{i+1}/{len(ROOMS)}] ", end="")
        try:
            capture_room(room)
        except Exception as e:
            print(f"Error on room {room}: {e}")

    print()
    print("Done! Screenshots saved to:")
    print("~/Library/Application Support/ScummVM/screenshots/")


if __name__ == "__main__":
    main()
