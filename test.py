import os

# 獲取 GITHUB_ENV 文件的路徑
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if GITHUB_TOKEN:
    print(true)
else
    print(false)
    
