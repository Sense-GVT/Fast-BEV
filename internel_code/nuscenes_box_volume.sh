WORLD_SIZE=$1
for i in `seq $WORLD_SIZE`; do
    let rank=$i-1
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    spring.submit run -p AD_GVT -n1 --job-name stat_$rank \
        --exclude=SH-IDC1-10-5-38-[33,41,148,170,174,201,220,233,236,41,80-83],SH-IDC1-10-5-39-[10] \
        "python internal_code/nuscenes_box_volume.py \
         --world_size $WORLD_SIZE \
         --rank $rank" \
    2>&1 | tee logs/stat_$rank.log > /dev/null &
done
