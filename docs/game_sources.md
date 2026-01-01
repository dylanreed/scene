# Game Sources for Training Data

## LucasArts SCUMM Games (VGA Era)

| Game | Year | Resolution | Notes |
|------|------|------------|-------|
| The Secret of Monkey Island | 1990 | 320x200 | Classic |
| Loom | 1990 | 320x200 | VGA version |
| Monkey Island 2: LeChuck's Revenge | 1991 | 320x200 | Classic |
| Indiana Jones and the Fate of Atlantis | 1992 | 320x200 | Classic |
| Day of the Tentacle | 1993 | 320x200 | Cartoon style |
| Sam & Max Hit the Road | 1993 | 320x200 | Cartoon style |
| Full Throttle | 1995 | 320x200 | Darker style |
| The Dig | 1995 | 320x200 | Sci-fi |
| Indiana Jones and the Last Crusade | 1989 | 320x200 | VGA version |
| Zak McKracken | 1988 | 320x200 | FM Towns VGA |

## Sierra SCI1 Games - Not Yet Extracted

| Game | Year | Status |
|------|------|--------|
| Leisure Suit Larry 1 VGA | 1991 | Don't have |
| Leisure Suit Larry 6 (floppy) | 1993 | Have - missing files |
| King's Quest 5 (floppy) | 1990 | Have - format issues |
| King's Quest 6 (floppy) | 1992 | Have - missing files |
| Space Quest 4 (floppy) | 1991 | Have - no resources |
| Space Quest 5 | 1993 | Have - missing files |
| Quest for Glory 3 | 1992 | Don't have |
| EcoQuest 2 | 1993 | Have - missing files |
| Mixed-Up Mother Goose VGA | 1991 | Don't have |
| Castle of Dr. Brain | 1991 | Don't have |
| Island of Dr. Brain | 1992 | Don't have |
| Freddy Pharkas | 1993 | Don't have |
| Pepper's Adventures in Time | 1993 | Have - missing files |
| Codename: ICEMAN | 1989 | Have - format issues |

## Sierra SCI1 Games - Already Extracted

| Game | Images |
|------|--------|
| Leisure Suit Larry 5 | 429 |
| Police Quest 1 VGA | 419 |
| Police Quest 3 | 395 |
| Conquests of the Longbow | 381 |
| Space Quest 1 VGA | 356 |
| EcoQuest 1 | 293 |

## Partially Extracted (PICs only, need SCI1.1 VIEW fix)

| Game | PICs |
|------|------|
| Police Quest 4 | 119 |
| Laura Bow 2 | 90 |
| Quest for Glory 1 VGA | 84 |
| Quest for Glory 4 | 58 |

## Notes

- **SCI1 format**: Sierra games from 1990-1993, uses RESOURCE.MAP + RESOURCE.000-007
- **SCUMM format**: LucasArts games, requires separate parser (not yet implemented)
- **SCI1.1 format**: Later Sierra games (1992+), VIEW resources use different header format
- All games target 320x200 VGA resolution, perfect for consistent training data

## Extraction Tools

- Sierra SCI1: `src/extract_all_games.py` (implemented)
- LucasArts SCUMM: Not yet implemented (ScummVM tools available)
