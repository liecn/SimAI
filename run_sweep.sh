#!/bin/bash

# Check if the number of arguments is less than 3
if [ "$#" -lt 3 ]; then
    # Print an error message to standard error
    echo "Error: You must provide all three arguments." >&2

    # Print the correct usage format
    echo "Usage: $0 [flowsim | ns3 | m4] [N] [M]" >&2

    # Exit the script with an error code
    exit 1
fi

N=$2
M=$3
TOPOFILE="topo_N${N}_M${M}.txt"

case "$1" in
  flowsim)
    time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 AS_FWIN=$((120 / M)) ./bin/SimAI_flowsim -w ./example/sweep/microAllReduce.txt -n ./example/sweep/$TOPOFILE -o ./results/flowsim_${N}_${M}/
    ;;
  ns3)
    # time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 AS_FWIN=$((400 / M)) ./bin/SimAI_simulator -t 8 -w ./example/sweep/microAllReduce.txt -n ./example/sweep/$TOPOFILE -c ./example/sweep/SimAI.conf -o ./results/ns3_${N}_${M}/ -r
    time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 AS_FWIN=100 ./bin/SimAI_simulator -t 8 -w ./example/sweep/microAllReduce.txt -n ./example/sweep/$TOPOFILE -c ./example/sweep/SimAI.conf -o ./results/ns3_${N}_${M}/ -r
    ;;
  m4)
    # time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 AS_FWIN=$((200 / M)) ./bin/SimAI_m4 -w ./example/sweep/microAllReduce.txt -n ./example/sweep/$TOPOFILE -o ./results/m4_${N}_${M}/
    time AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_m4 -w ./example/sweep/microAllReduce.txt -n ./example/sweep/$TOPOFILE -o ./results/m4_${N}_${M}/
    ;;
  *) # This is a catch-all for any other value
    echo "Error: Invalid simulator '$1'. Please choose 'flowsim', 'ns3', or 'm4'."
    exit 1
    ;;
esac

echo "Script finished."
