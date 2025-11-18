local_ip=x.x.x.x

export HCCL_IF_IP=$local_ip         # 指定HCCL通信库使用的网卡 IP 地址
export GLOO_SOCKET_IFNAME=$nic_name # 指定使用 Gloo通信库时指定网络接口名称 
export TP_SOCKET_IFNAME=$nic_name   # 指定 TensorParallel使用的网络接口名称
export HCCL_SOCKET_IFNAME=$nic_name # 指定 HCCL 通信库使用的网络接口名称
export OMP_PROC_BIND=false          # 允许操作系统调度线程在多个核心之间迁移
export OMP_NUM_THREADS=100          # 在支持 OpenMP 的程序中，最多使用 100 个 CPU 线程进行并行计算
export VLLM_USE_V1=1                # 强制使用v1模型加载/推理路径
export HCCL_BUFFSIZE=1024           # 每个通信操作的缓冲区大小为 1024 Bytes
# export VLLM_TORCH_PROFILER_DIR="/home/l00889328/profiling" # profiling保存路径
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_ASCEND_ENABLE_CP=1

vllm serve /mnt/weight/deepseek_diff/deepseek_r1_w8a8_vllm/  \
  --host 0.0.0.0 \
  --port 8004 \
  --served-model-name deepseek_r1 \
  --data-parallel-size 1 \
  --data-parallel-size-local 1 \
  --data-parallel-address x.x.x.x \
  --data-parallel-rpc-port 13389 \
  --tensor-parallel-size 8 \
  --context-parallel-size 2 \
  --decode_context_parallel_size 8 \
  --enable-expert-parallel \
  --no-enable-prefix-caching \
  --max-num-seqs 1 \
  --max-model-len 4196 \
  --max-num-batched-tokens 6020 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --enforce-eager \
  --quantization ascend \
  --additional-config '{"ascend_scheduler_config":{"enabled":false},"torchair_graph_config":{"enabled":false, "enable_multistream_moe":false, "use_cached_graph":false}}'
