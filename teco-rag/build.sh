# 清理构建的临时文件
rm -rf build
rm -rf dist
rm -rf *.egg-info

echo "========================================================"
echo "build temp files removed"
echo "========================================================"

# Python工程构建打包
python3 setup.py sdist --formats=gztar
echo "========================================================"
echo "Python package success"
echo "========================================================"

# 登录镜像仓库
#docker login --username=xxxxxx -p 123456 url

APP_VERSION=1.0.0
TIMESTAMP=`date +%Y%m%d%H%M%S`

docker build --pull \
    --network=host \
    --build-arg APP_VERSION=$APP_VERSION \
    -t teco-rag:$APP_VERSION \
    -f APP-META/Dockerfile .
