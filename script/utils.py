import logging
import os
import sys

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建文件处理器
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/script.log", mode="a")
file_handler.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.ERROR)

# 创建格式化器
formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(thread)d - %(levelname)s : %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)
