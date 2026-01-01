-- AppleScript to automate ScummVM room captures
-- Run with: osascript src/capture_scummvm.scpt

set roomList to {100, 105, 110, 111, 115, 120, 130, 140, 145, 150, 160, 170, 180, 190, 200, 215, 220, 225, 230, 235, 240, 250, 260, 270, 280, 290, 310, 315, 320, 330, 335, 340, 345, 350, 355, 370, 375, 380, 385, 390, 400, 405, 410, 415, 420, 425, 430, 440, 450, 460, 465, 470, 480, 500, 510, 520, 525, 530, 535, 600, 610, 620, 630, 640, 650, 660, 670, 690, 700, 710, 720, 730, 738, 740, 750, 760, 780, 790, 795, 800, 810, 820, 840, 850, 860, 870, 880, 890, 900, 905, 910, 915, 920}

-- Activate ScummVM window
tell application "System Events"
    set frontmost of process "scummvm" to true
end tell

delay 1

-- Open debugger with Ctrl+D
tell application "System Events"
    tell process "scummvm"
        key code 2 using control down -- D key
    end tell
end tell

delay 0.5

repeat with roomNum in roomList
    -- Type room command
    tell application "System Events"
        tell process "scummvm"
            keystroke "room " & roomNum
            key code 36 -- Enter
        end tell
    end tell

    delay 0.3

    -- Type go command
    tell application "System Events"
        tell process "scummvm"
            keystroke "go"
            key code 36 -- Enter
        end tell
    end tell

    delay 1 -- Wait for room to render

    -- Take screenshot with Alt+S
    tell application "System Events"
        tell process "scummvm"
            key code 1 using option down -- S key
        end tell
    end tell

    delay 0.5

    -- Open debugger again for next room
    tell application "System Events"
        tell process "scummvm"
            key code 2 using control down -- D key
        end tell
    end tell

    delay 0.3
end repeat

display dialog "Capture complete! Check ~/Library/Application Support/ScummVM/screenshots/"
