# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:18:43 2026

@author: cns-th-lab
"""
import keyboard
import csv
import time
import sys

# ================= Configuration =================
RATS = ["198", "199", "234", "235", "237", "238", "274", "400", "402", "419", "421", "422", "424", "483"]
OUTPUT_FILE = "behavior_counts.csv"

# List your behaviors here (up to 10 to map to 1-0)
BEHAVIORS = [
    "Poke Center",
    "Poke Left",
    "Poke Right",
    "Walk Forward",
    "Walk Left",
    "Walk Right",
    "Grooming(curled)",
    "Grooming(ears)",
    "Stand Up",
    "Turn Left",
    "Turn Right",
    "Haunches",
    ]
# =================================================

def main():
    # Map keys '1'-'9' and '0' to the behaviors list
    keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b']
    key_map = {keys[i]: BEHAVIORS[i] for i in range(len(BEHAVIORS))}
    
    results = []

    print("--- Rat Behavior Logger ---")
    print("Controls:")
    for k, v in key_map.items():
        print(f"  [{k}] - {v}")
    print("  [Backspace] - Undo last entry")
    print("  [Enter]     - Finish current rat / Next rat")
    print("  [Esc]       - Quit program immediately")
    print("-" * 27)

    for rat in RATS:
        print(f"\nCurrently scoring: {rat}")
        
        # Initialize counts and history for the current rat
        counts = {b: 0 for b in BEHAVIORS}
        history = []

        time.sleep(0.2) # Prevent carrying over the Enter keypress from the terminal

        while True:
            event = keyboard.read_event()
            
            # Only trigger when the key is pressed down (ignores key release)
            if event.event_type == keyboard.KEY_DOWN:
                key = event.name

                if key in key_map:
                    behavior = key_map[key]
                    counts[behavior] += 1
                    history.append(behavior)  # Add to history for undo
                    print(f" [+] {behavior:10} | Total: {counts[behavior]}")

                elif key == 'backspace':
                    # Undo functionality
                    if history:
                        last_behavior = history.pop()
                        counts[last_behavior] -= 1
                        print(f" [-] UNDO: Removed 1 {last_behavior} | Total: {counts[last_behavior]}")
                    else:
                        print(" [!] History is empty. Nothing to undo.")

                elif key == 'enter':
                    print(f" [v] Finished scoring {rat}.")
                    break

                elif key == 'esc':
                    print("\n[!] Exiting program. Saving progress...")
                    save_to_csv(results)
                    sys.exit()

                # Small delay to prevent a single long press from registering multiple times
                time.sleep(0.15)

        # Build the dictionary row for the CSV
        row = {'Rat': rat}
        row.update(counts)
        results.append(row)

    # Save once all rats are processed
    save_to_csv(results)


def save_to_csv(data):
    if not data:
        print("No data to save.")
        return
        
    fieldnames = ['Rat'] + BEHAVIORS
    try:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"\nData successfully saved to {OUTPUT_FILE}")
    except PermissionError:
        print(f"\n[Error] Close {OUTPUT_FILE} if you have it open and run again.")

if __name__ == "__main__":
    main()