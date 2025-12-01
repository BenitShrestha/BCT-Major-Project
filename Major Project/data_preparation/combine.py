import glob
import os

# Get all .txt files in the current directory
input_files = sorted(glob.glob("*.txt"))  # Automatically sorts by filename

output_file = "combined_output.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    for filename in input_files:
        with open(filename, "r", encoding="utf-8") as infile:
            outfile.write(infile.read())
            outfile.write("\n")  # Optional: add newline between files
        print(f"Merged: {filename}")

print(f"\n✅ All files merged into: {output_file}")
