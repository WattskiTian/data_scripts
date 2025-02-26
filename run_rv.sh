SE_PATH=/home/watts/gem5/configs/deprecated/example/se.py
OUT_PATH=/home/watts/dhrystone/gem5output_rv
ARCH=Riscv
# CPU_TYPE=TimingSimpleCPU
CPU_TYPE=AtomicSimpleCPU
# DEBUG_FLAG=ExecAll,-ExecFaulting,-ExecSymbol,-ExecThread,-ExecAsid,-ExecMicro,FmtTicksOff,IntRegs,MyDebugTest
DEBUG_FLAG=ExecAll,-ExecSymbol,-ExecThread,-ExecAsid,-ExecMicro,FmtTicksOff,IntRegs,MyDebugTest
# DEBUG_FLAG=MyDebugTest
./gem5_RV.opt --outdir=$OUT_PATH \
  --debug-flags=$DEBUG_FLAG \
  --debug-file=$OUT_PATH/debug_log \
  $SE_PATH \
  --cpu-type=$ARCH$CPU_TYPE \
  --mem-size=8196MB \
  --maxinsts=1000000 \
  --cmd=./bench/bench1_rv32
  
python3 trace_dealer.py
python3 fronted_trace.py
# python3 load.py
# # python3 load_plt.py
# python3 model_data.py
# python3 ghis.py
# python3 random_forest.py
# python3 rnn.py
# python3 data_dealer.py
