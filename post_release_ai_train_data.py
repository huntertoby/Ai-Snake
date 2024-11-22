import os
import requests
from datetime import datetime
import pytz

# 獲取 GITHUB_TOKEN
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "huntertoby/Ai-Snake"  # 替換為你的儲存庫名稱

if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN is not set. Please check the environment variables.")

print(f"Using GITHUB_TOKEN: {GITHUB_TOKEN[:4]}***")  # 僅顯示部分 Token 用於調試

def get_readme():
    url = f"https://api.github.com/repos/{REPO}/contents/README.md"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch README.md: {response.status_code}, {response.json()}")
        return None

def update_readme(content, sha):
    url = f"https://api.github.com/repos/{REPO}/contents/README.md"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "message": "Update README with new highest score",
        "content": content,
        "sha": sha,
    }
    response = requests.put(url, json=data, headers=headers)
    if response.status_code == 200:
        print("README.md updated successfully.")
    else:
        print(f"Failed to update README.md: {response.status_code}, {response.json()}")

def create_release(tag_name, release_name, description):
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
        print("Release created successfully.")
        release = response.json()
        print(f"Release URL: {release['html_url']}")
    else:
        print(f"Failed to create release: {response.status_code}, {response.json()}")

if __name__ == "__main__":
    # 設定台灣時區
    tz = pytz.timezone("Asia/Taipei")
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
    tag = f"training-results-{datetime.now(tz).strftime('%Y%m%d-%H%M')}"
    release_name = f"AI Snake Training Results ({current_time})"

    # 讀取 train_results.txt 的內容作為 Release 描述
    result_file = "train_results.txt"
    if os.path.exists(result_file):
        with open(result_file, "r") as file:
            description = file.read()
            highest_score = max(int(line.split("Highest Score: ")[1].split(" |")[0])
                                for line in description.splitlines() if "Highest Score" in line)
    else:
        description = "No training results available. File not found."
        highest_score = 0

    # 獲取 README 文件並檢查最高分
    readme = get_readme()
    if readme:
        readme_content = readme["content"]
        readme_sha = readme["sha"]

        # 解碼現有內容
        import base64
        decoded_content = base64.b64decode(readme_content).decode("utf-8")

        # 檢查是否需要更新最高分
        if f"Highest Score: {highest_score}" not in decoded_content:
            new_content = decoded_content + f"\n\n## Highest Score\nAchieved {highest_score} points on {current_time}."
            encoded_content = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")
            update_readme(encoded_content, readme_sha)

    # 創建 Release
    create_release(tag, release_name, description)
