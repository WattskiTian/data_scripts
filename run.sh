SE_PATH=/home/watts/gem5/configs/deprecated/example/se.py
# SE_PATH=./simple_se.py
./gem5.opt --outdir=/home/watts/dhrystone/gem5output \
  --debug-flags=Exec --debug-file=/home/watts/dhrystone/gem5output/debug_log \
  $SE_PATH \
  --cpu-type=X86AtomicSimpleCPU \
  --maxinsts=100 \
  --cmd=./dry
