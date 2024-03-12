date=$(date '+%Y-%m-%d-%H:%M:%S')

mkdir -p log

dir=ops
name=test_scan

pytest ${dir}/${name}.py  2>&1 | tee -a log/${date}-${name}.log