# 配置环境变量
MPI_HOME=/usr/local/openmpi-4.0.1 
PATH=$MPI_HOME/bin:$PATH 
LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH


FT_CB_BATCH_SIZE=14 FT_PAGE_ATTENTION=ON FT_CB_MAX_INPUT_LEN=4096 FT_CB_MAX_OUTPUT_LEN=1024 /opt/tritonserver/bin/tritonserver --model-repository=/mnt/nvme/Qwen-7B-Chat/model_configs/Qwen-7B-Chat
# FT_CB_BATCH_SIZE 为并发数量，请根据实际需求进行设置