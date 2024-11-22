import os
import requests
from datetime import datetime
import pytz
import base64

# 獲取 GITHUB_TOKEN
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "huntertoby/Ai-Snake"  # 替換為你的儲存庫名稱

if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN is not set. Please check the environment variables.")

def get_readme():
    """獲取 README.md 文件內容"""
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
    """更新 README.md 文件"""
    url = f"https://api.github.com/repos/{REPO}/contents/README.md"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {
        "message": "Update README with latest training results",
        "content": content,
        "sha": sha,
    }
    response = requests.put(url, json=data, headers=headers)
    if response.status_code == 200:
        print("README.md updated successfully.")
    else:
        print(f"Failed to update README.md: {response.status_code}, {response.json()}")

def extract_training_count(content):
    """從 README 內容中提取訓練次數"""
    for line in content.splitlines():
        if line.startswith("現在已經訓練了:"):
            count_str = line.split(": **")[1].split("**")[0].strip()
            return int(count_str)
    return 0

if __name__ == "__main__":
    # 設定台灣時區
    tz = pytz.timezone("Asia/Taipei")
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    # 訓練結果檔案
    result_file = "train_results.txt"

    if os.path.exists(result_file):
        with open(result_file, "r") as file:
            results = file.readlines()
            # 提取最高分和對應資訊
            best_result = max(results, key=lambda x: int(x.split("Score: ")[1].split(" |")[0]))
            episode = best_result.split("Episode ")[1].split(" |")[0]
            score = best_result.split("Score: ")[1].split(" |")[0]
            epsilon = best_result.split("Epsilon: ")[1].strip()
    else:
        print("No training results available. File not found.")
        exit()

    # 獲取 README 文件
    readme = get_readme()
    if readme:
        readme_content = base64.b64decode(readme["content"]).decode("utf-8")
        readme_sha = readme["sha"]

        # 提取目前總訓練次數，並加上 1000
        current_training_count = extract_training_count(readme_content)
        updated_training_count = current_training_count + 1000

        # 更新 README 格式
        new_content = f"""
# AI Snake Project

## **最佳成績**
現在使用 GitHub Action 訓練最佳成績為在 10x10 的地圖情況下  
(台灣時間) 第 **{episode}** 次訓練  
**分數**: {score}  
**探索值**: {epsilon}

## 總訓練次數
現在已經訓練了: **{updated_training_count}** 次
"""

        # 編碼新的 README 文件內容
        encoded_content = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")
        update_readme(encoded_content, readme_sha)
