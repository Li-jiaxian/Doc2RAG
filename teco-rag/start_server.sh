#!/bin/bash
source .venv/bin/activate
# 切换到正确的工作目录
# cd "$(dirname "$0")/teco-rag"

# 加载环境变量
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "已加载.env文件中的环境变量"
else
    echo "警告: .env文件不存在"
fi

# 启动 Server
python3 server/main.py --host 0.0.0.0 --create_tables
