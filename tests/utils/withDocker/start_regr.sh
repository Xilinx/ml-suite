# Build xfdnn lib
ssh 10.22.64.112 "cd /wrk/acceleration/users/anup/MLsuite_mastr/tests/utils/withDocker && bash -x build_lib.sh"

# Running on FPGA
ssh xsjfislx18 "docker exec mluser_container /bin/sh "navigate.sh""


# trigger mail
ssh xsjfislx12 "cd /wrk/acceleration/users/anup/MLsuite_mastr/tests/utils/withDocker && bash -x mail.sh"

# Copy log files
cur_date=$(date +"%d%b%y")
mv output_logs_${cur_date} /wrk/acceleration/test_deephi/daily_logs/output_logs_${cur_date}_Docker

# Remove one week older folder to free up space
find /wrk/acceleration/test_deephi/daily_logs/* -type d -ctime +6 -exec rm -rf {} +


