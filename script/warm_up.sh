# 先确保已安装 teco-client-toolkits包
# pip install http://mirrors.tecorigin.com/repository/teco-pypi-repo/packages/teco-client-toolkits/0.0.1/teco_client_toolkits-0.0.1-py3-none-any.whl
python concurrency_warm_up.py --ip 127.0.0.1 --port 8001 --qps 1
# <qps> 请设置为 >= FT_CB_BATCH_SIZE的正整数