date=$(date '+%Y-%m-%d-%H:%M:%S')

mkdir -p log

name=mnet_pytorch

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 pytest ${name}.py  2>&1 | tee -a log/${date}-${name}.log
# CUDA_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 python ${name}.py  2>&1 | tee -a log/${date}-${name}.log