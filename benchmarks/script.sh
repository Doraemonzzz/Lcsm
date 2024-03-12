date=$(date '+%Y-%m-%d-%H:%M:%S')

mkdir -p log

name=benchmark_scan

python ${name}.py  2>&1 | tee -a log/${date}-${name}.log
