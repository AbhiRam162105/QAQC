import zipfile
import os

# Path to your zip file
zip_path = '/root/QAQC/Archive.zip'

# Directory to extract to
extract_to = '/root/QAQC/'

# Ensure the output directory exists
os.makedirs(extract_to, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Extracted '{zip_path}' to '{extract_to}'")