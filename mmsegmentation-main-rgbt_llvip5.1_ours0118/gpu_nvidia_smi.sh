# 快速检查所有GPU节点的显存情况
for node in $(sinfo -N -p cluster02 -o "%N" | grep gpu); do
    echo "--- $node ---"
    srun -w $node -n1 -N1 --time=00:01:00 nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv 2>/dev/null | head -2
    echo
done
