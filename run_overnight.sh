#!/bin/bash
# Overnight run script with logging and crash recovery

LOGFILE="run_$(date +%Y%m%d_%H%M%S).log"
SCRIPT="run_llm_eval.py"

echo "Starting at $(date)" | tee -a "$LOGFILE"
echo "Log file: $LOGFILE"

# Activate conda environment and run
source ~/miniforge3/etc/profile.d/conda.sh
conda activate green-llm-eval

# Run with nohup to survive terminal close, unbuffered output for real-time logging
nohup python -u "$SCRIPT" >> "$LOGFILE" 2>&1 &

PID=$!
echo "Started with PID: $PID" | tee -a "$LOGFILE"
echo "$PID" > run.pid

echo ""
echo "=== Commands ==="
echo "Watch progress:  tail -f $LOGFILE"
echo "Check if running: ps -p $PID"
echo "Stop:            kill $PID"
