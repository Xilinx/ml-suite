#!/bin/bash

Time_Stamp=`date +%Y%m%d_%H%M%S`
for device_id in {0..7}
do
    echo >> log_${Time_Stamp}
    echo "Test device ${device_id}..." >> log_${Time_Stamp}
    echo >> log_${Time_Stamp}

    export XBLAS_DEVICE_IDX=${device_id}
    ./run.sh 1526 test_classify large 8 >> log_${Time_Stamp}
done
