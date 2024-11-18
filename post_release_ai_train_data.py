import os
import requests

# 配置 GitHub Token 和儲存庫名稱
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # 設定為 GitHub Actions 的 secret
REPO = "huntertoby/Ai-Snake"  # 替換為你的儲存庫名稱

def create_release(tag_name, release_name, description, file_path):
    url = f"https://api.github.com/repos/{REPO}/releases"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "tag_name": tag_name,
        "name": release_name,
        "body": description,
        "draft": False,
        "prerelease": False,
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        release = response.json()
        upload_url = release["upload_url"].split("{")[0]
        upload_asset(upload_url, file_path)
    else:
        print(f"Failed to create release: {response.json()}")

def upload_asset(upload_url, file_path):
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "text/plain",
    }
    params = {"name": os.path.basename(file_path)}

    with open(file_path, "rb") as file:
        response = requests.post(upload_url, headers=headers, params=params, data=file)
        if response.status_code == 201:
            print(f"Successfully uploaded {file_path} to release.")
        else:
            print(f"Failed to upload asset: {response.json()}")

if __name__ == "__main__":
    tag = "training-results"
    release_name = "AI Snake Training Results"
    description = "This release contains the training results for AI Snake."
    result_file = "train_results.txt"

    if os.path.exists(result_file):
        create_release(tag, release_name, description, result_file)
    else:
        print(f"{result_file} does not exist.")
