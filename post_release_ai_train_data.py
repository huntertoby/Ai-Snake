import os
import requests
from datetime import datetime

# 獲取 GITHUB_TOKEN
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "huntertoby/Ai-Snake"  # 替換為你的儲存庫名稱

if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN is not set. Please check the environment variables.")

print(f"Using GITHUB_TOKEN: {GITHUB_TOKEN[:4]}***")  # 僅顯示部分 Token 用於調試

def create_release(tag_name, release_name, description):
    url = f"https://api.github.com/repos/{REPO}/releases"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "tag_name": tag_name,
        "name": release_name,  # 使用動態標題
        "body": description,   # 文件內容作為描述
        "draft": False,
        "prerelease": False,
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        print("Release created successfully.")
        release = response.json()
        print(f"Release URL: {release['html_url']}")
    else:
        print(f"Failed to create release: {response.status_code}, {response.json()}")

if __name__ == "__main__":
    # 動態生成標題和標籤
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = f"training-results-{datetime.now().strftime('%Y%m%d')}"
    release_name = f"AI Snake Training Results ({current_time})"

    # 讀取 train_results.txt 的內容作為 Release 描述
    result_file = "train_results.txt"
    if os.path.exists(result_file):
        with open(result_file, "r") as file:
            description = file.read()  # 讀取文件內容
    else:
        description = "No training results available. File not found."

    create_release(tag, release_name, description)
