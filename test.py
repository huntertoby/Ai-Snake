import os

# 獲取 GITHUB_TOKEN
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if GITHUB_TOKEN:
    print("GITHUB_TOKEN is available.")
else:
    print("GITHUB_TOKEN is not available. Please check the configuration.")
