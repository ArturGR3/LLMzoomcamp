import os
import requests

def download_github_folder(repo_url, folder_path, local_dir, token=None):
    # Extract user/repo from URL
    repo_name = repo_url.strip('/').split('/')[-2:]
    repo_name = '/'.join(repo_name)

    # Set the API URL
    api_url = f"https://api.github.com/repos/{repo_name}/contents/{folder_path}"

    # Headers for authorization if a token is provided (for private repos or higher rate limits)
    headers = {}
    if token:
        headers['Authorization'] = f'token {token}'

    # Get the folder contents
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        folder_contents = response.json()
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        for item in folder_contents:
            if item['type'] == 'file':
                file_url = item['download_url']
                file_path = os.path.join(local_dir, item['name'])
                download_file(file_url, file_path)
    else:
        print(f"Error: {response.status_code}, {response.text}")

def download_file(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Error downloading {url}: {response.status_code}, {response.text}")

# Example usage
repo_url = "https://github.com/DataTalksClub/llm-zoomcamp"
folder_path = "04-monitoring/app"
local_dir = "hw4/app"

download_github_folder(repo_url, folder_path, local_dir)
