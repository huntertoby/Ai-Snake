import os
import requests
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import base64

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "huntertoby/Ai-Snake"

if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN is not set. Please check the environment variables.")

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
        "message": "Update README with latest training results",
        "content": content,
        "sha": sha,
    }
    response = requests.put(url, json=data, headers=headers)
    if response.status_code == 200:
        print("README.md updated successfully.")
    else:
        print(f"Failed to update README.md: {response.status_code}, {response.json()}")

def extract_training_count_and_best_len(content):
    training_count = 0
    best_len = 0
    for line in content.splitlines():
        if line.startswith("現在已經訓練了:"):
            training_count = int(line.split(": **")[1].split("**")[0].strip())
        if line.startswith("**長度**:"):
            best_len = int(float(line.split(": ")[1].strip()))
    return training_count, best_len

def create_release(tag_name, release_name, description, asset_paths):
    """創建 GitHub Release 並上傳圖表和其他資產"""
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
        upload_url = release["upload_url"].replace("{?name,label}", "")

        for asset_path in asset_paths:
            with open(asset_path, "rb") as file:
                headers["Content-Type"] = "application/octet-stream"
                upload_response = requests.post(
                    f"{upload_url}?name={os.path.basename(asset_path)}",
                    headers=headers,
                    data=file
                )
            if upload_response.status_code == 201:
                print(f"Asset uploaded successfully: {asset_path}")
            else:
                print(f"Failed to upload asset: {upload_response.status_code}, {upload_response.json()}")
    else:
        print(f"Failed to create release: {response.status_code}, {response.json()}")

def plot_results(results):
    episodes = [result["episode"] for result in results]
    lengths = [result["Len"] for result in results]
    epsilons = [result["epsilon"] for result in results]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, lengths, label="Length", marker="o", linestyle="")  
    plt.plot(episodes, epsilons, label="Epsilon", marker="x", linestyle="")  
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title("Training Results: Length and Epsilon over Episodes")
    plt.legend()
    plt.grid(True)

    output_path = "training_results.png"
    plt.savefig(output_path)
    print(f"Plot saved as {output_path}")
    return output_path

if __name__ == "__main__":
    tz = pytz.timezone("Asia/Taipei")
    current_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    result_file = "train_results.txt"

    if os.path.exists(result_file):
        with open(result_file, "r") as file:
            results = []
            for line in file:
                parts = line.strip().split(" | ")
                episode = int(parts[0].split("Episode ")[1])
                length = int(parts[1].split("Len: ")[1])
                longest_len = int(parts[2].split("Longest Len: ")[1])
                epsilon = float(parts[3].split("Epsilon: ")[1])
                results.append({
                    "episode": episode,
                    "Len": length,
                    "LongestLen": longest_len,
                    "epsilon": epsilon
                })
            best_result = max(results, key=lambda x: x["Len"])
            episode = best_result["episode"]
            length = best_result["Len"]
            epsilon = best_result["epsilon"]
    else:
        print("No training results available. File not found.")
        exit()

    readme = get_readme()
    if readme:
        readme_content = base64.b64decode(readme["content"]).decode("utf-8")
        readme_sha = readme["sha"]

        current_training_count, current_best_len = extract_training_count_and_best_len(readme_content)

        updated_training_count = current_training_count + 1000

        if length > current_best_len:
            new_content = f"""
# AI Snake Project

## **最佳成績**
現在使用 GitHub Action 訓練最佳成績為在 10x10 的地圖情況下  
(時間 {current_time}) 第 **{episode}** 次訓練  
**長度**: {length}  
**探索值**: {epsilon}

## 總訓練次數
現在已經訓練了: **{updated_training_count}** 次
"""
        else:
            new_content = f"""
# AI Snake Project

## **最佳成績**
{readme_content.split("## **最佳成績**")[1].split("## 總訓練次數")[0]}

## 總訓練次數
現在已經訓練了: **{updated_training_count}** 次
"""

        encoded_content = base64.b64encode(new_content.encode("utf-8")).decode("utf-8")
        update_readme(encoded_content, readme_sha)

    chart_path = plot_results(results)
    tag_name = f"training-results-{datetime.now(tz).strftime('%Y%m%d-%H%M')}"
    release_name = f"AI Snake Training Results ({current_time})"
    release_description = f"""
### 訓練結果
- **最佳長度**: {length}
- **探索值**: {epsilon}
- **訓練次數**: {updated_training_count}
    """
    asset_paths = [chart_path, result_file, "highest_score_game_state.png"]
    create_release(tag_name, release_name, release_description, asset_paths)
