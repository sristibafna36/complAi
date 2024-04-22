import os
import shutil

# List of filenames you want to move
filenames = ["1242.pdf","1532.pdf","1742.pdf","2320.pdf","2356.pdf","2401.pdf","3054.pdf","3659.pdf","3704.pdf","4285.pdf","4318.pdf","4354.pdf","5072.pdf","5494.pdf","5804.pdf","6509.pdf","6666.pdf","6980.pdf","7321.pdf","7417.pdf","7421.pdf","8098.pdf","8975.pdf","8998.pdf","9002.pdf","9775.pdf","9909.pdf","10577.pdf","10782.pdf","11029.pdf","11314.pdf","11645.pdf","11967.pdf","12054.pdf","12189.pdf","12525.pdf"]

# Source directory
source_dir = 'E:\Bulb\docs'

# Destination directory
destination_dir = 'E:\Bulb\docs\Sample'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Move each file
for filename in filenames:
    source_path = os.path.join(source_dir, filename)
    destination_path = os.path.join(destination_dir, filename)

    # Check if the file exists in the source directory
    if os.path.exists(source_path):
        # Move the file
        shutil.move(source_path, destination_path)
        print(f"Moved: {filename}")
    else:
        print(f"File not found: {filename}")
