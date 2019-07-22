#!/bin/bash

# Export MLSuite path
if [ -z $MLSUITE_ROOT ];
then
    export MLSUITE_ROOT=/wrk/acceleration/users/anup/MLsuite_mastr
    echo "##### Setting default path as : $MLSUITE_ROOT. Please set to required path"
    #exit 1;
fi

# Build xfdnn lib
ssh 10.22.64.112 "cd ${MLSUITE_ROOT}/tests/utils/withDocker && bash -x build_lib.sh"


# Running on FPGA
ssh xsjsda87 "cd ${MLSUITE_ROOT}/tests/utils/withDocker && bash -x docker_run.sh"
ssh xsjsda87 "docker exec mluser_container /bin/sh "tests/utils/withDocker/navigate.sh""
ssh xsjsda87 "docker exec mluser_container /bin/sh "tests/utils/withDocker/en_permissions.sh""

################## Table Generation
python gen_table.py nw_status.txt

cur_date=$(date +"%d%b%y")

mv nw_status.txt output_logs/nw_status_$cur_date.txt
mv xfdnn_nightly.csv output_logs/xfdnn_nightly_$cur_date.csv
mv output_logs output_logs_$cur_date


# trigger mail
ssh xsjfislx12 "cd ${MLSUITE_ROOT}/tests/utils/withDocker && bash -x mail.sh"

# Copy log files
mv output_logs_${cur_date} /wrk/acceleration/test_deephi/daily_logs/output_logs_${cur_date}_Docker

# Remove one week older folder to free up space
find /wrk/acceleration/test_deephi/daily_logs/* -type d -ctime +6 -exec rm -rf {} +


