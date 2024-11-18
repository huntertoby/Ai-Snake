import os

# 從環境變數中讀取 secret

secret = os.getenv.get("AI_SNAKE_ACTIONS_SECRETS")

if secret:
    print(f"The secret is: {secret}")
else:
    print("The secret is not available. Please check the configuration.")
