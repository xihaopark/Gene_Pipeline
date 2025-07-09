# api.example.py
# API密钥配置模板文件
# 使用方法：
# 1. 复制此文件为 api.py
# 2. 将下面的占位符替换为实际的API密钥

# Your Claude API key
ANTHROPIC_API_KEY = "your_anthropic_api_key_here"

# 获取API密钥的方法：
# Claude API: https://console.anthropic.com/
# 
# 示例cURL命令：
# curl https://api.anthropic.com/v1/messages \
#         --header "x-api-key: YOUR_API_KEY" \
#         --header "anthropic-version: 2023-06-01" \
#         --header "content-type: application/json" \
#         --data \
#     '{
#         "model": "claude-3-5-sonnet-20241022",
#         "max_tokens": 1024,
#         "messages": [
#             {"role": "user", "content": "Hello, world"}
#         ]
#     }' 