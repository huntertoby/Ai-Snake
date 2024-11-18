import os

# 獲取 GITHUB_ENV 文件的路徑
env_file = os.getenv('GITHUB_ENV')

# 寫入新的環境變數
with open(env_file, "a") as myfile:
    myfile.write("MY_VAR=MY_VALUE\n")
