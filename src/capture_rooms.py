#!/usr/bin/env python3
"""
Capture room screenshots from ScummVM using save states.

This creates a ScummVM debug script that visits each room and saves screenshots.
Run ScummVM with: scummvm --debugflags=Scripts lsl5
Then paste the commands from the generated script file.
"""
import os

# Known room numbers from Larry 5 PIC resources
ROOMS = [
    100, 105, 110, 111, 115, 120, 130, 140, 145, 150,
    160, 170, 180, 190, 200, 215, 220, 225, 230, 235,
    240, 250, 258, 260, 270, 280, 290, 310, 315, 320,
    330, 335, 340, 345, 350, 355, 370, 375, 380, 385,
    390, 400, 405, 410, 415, 420, 425, 430, 440, 450,
    460, 465, 470, 480, 500, 510, 520, 525, 530, 535,
    600, 610, 620, 630, 640, 650, 660, 670, 690, 700,
    710, 720, 730, 738, 740, 750, 760, 780, 790, 795,
    800, 810, 820, 840, 850, 860, 870, 880, 890, 900,
    905, 910, 915, 920
]

def generate_debug_script():
    """Generate ScummVM debugger commands to capture all rooms."""
    script = []
    script.append("# ScummVM Debug Commands to capture room screenshots")
    script.append("# Paste these into the ScummVM debugger (Ctrl+D)")
    script.append("")

    for room in ROOMS:
        script.append(f"# Room {room}")
        script.append(f"room {room}")
        script.append("go")
        script.append("")

    return "\n".join(script)

def main():
    script = generate_debug_script()

    output_path = "extracts/scummvm_capture_commands.txt"
    os.makedirs("extracts", exist_ok=True)

    with open(output_path, "w") as f:
        f.write(script)

    print(f"Generated: {output_path}")
    print()
    print("Instructions:")
    print("1. Start ScummVM with Larry 5")
    print("2. Press Ctrl+D to open debugger")
    print("3. For each room, type: room <number>")
    print("4. Then type: go")
    print("5. Take screenshot with Alt+S (or Cmd+S on Mac)")
    print()
    print("Or use AppleScript to automate (see capture_rooms_auto.scpt)")

if __name__ == "__main__":
    main()
