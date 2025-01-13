import os
import requests

# Paths in WSL format
links_file_path = "/Users/kanoalindiwe/Downloads/temp/0_file_download_links.txt"
download_folder = "/Users/kanoalindiwe/Downloads/temp/lidar"

# Read the download links from the file
with open(links_file_path, 'r') as file:
    download_links = file.readlines()

# Loop through each link and download if the file doesn't already exist or is incomplete
for link in download_links:
    link = link.strip()  # Remove any surrounding whitespace
    if not link:
        continue  # Skip empty lines

    # Extract the filename from the link
    filename = os.path.basename(link)
    file_path = os.path.join(download_folder, filename)

    try:
        # Send a HEAD request to get the file size
        head_response = requests.head(link)
        head_response.raise_for_status()
        file_size = int(head_response.headers.get('Content-Length', 0))

        # Check if the file exists and if its size matches the expected size
        if os.path.exists(file_path):
            existing_file_size = os.path.getsize(file_path)
            if existing_file_size == file_size:
                print(f"File already fully downloaded, skipping: {file_path}")
                continue
            else:
                print(f"File is partially downloaded, re-downloading: {file_path}")

        # Download the file
        response = requests.get(link, stream=True)
        response.raise_for_status()

        # Save the file to the specified directory
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded: {file_path}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {link}: {e}")
