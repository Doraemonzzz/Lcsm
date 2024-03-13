date=$(date '+%Y-%m-%d-%H:%M:%S')

mkdir -p log

dir=ops
name=test_scan

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 pytest ${dir}/${name}.py  2>&1 | tee -a log/${date}-${name}.log
