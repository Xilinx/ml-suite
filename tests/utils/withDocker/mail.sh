
cur_date=$(date +"%d%b%y")

#MAIL_LIST=anup@xilinx.com
MAIL_LIST="asirasa@xilinx.com aaronn-all@xilinx.com elliott-all@xilinx.com sumitn-all@xilinx.com"
#MAIL_LIST_XHD="sumitn-all@xilinx.com"

Yesterday_date=$(date --date="-1 day" +"%Y-%m-%d")
today_date=$(date +"%Y-%m-%d")
yesterday_date1=$(date --date="-1 day" +"%d%b%y")

# Remove old files
#rm -rf git_commit_log.txt

# Capture commits from previous run
GIT_LOG=$(git log --pretty=format:"%H : %aD - %an: %s %n" --after="<Yesterday_date> 06:30")
#echo "$GIT_LOG" >> git_commit_log.txt
#git log  --after="Yesterday_date" 2>&1 | tee git_commit_log.txt


# Invoke mail
if [ -z "$GIT_LOG" ];
then
        echo -e "No commits made across MLSuite repo from $yesterday_date1 to $cur_date" | mailx -s "Git commit log - $cur_date" ${MAIL_LIST}
else
        echo -e "Below are the commits made across MLSuite repo from $yesterday_date1 to $cur_date.\n\n\n${GIT_LOG}" | mailx -s "Git commit log - $cur_date" ${MAIL_LIST}
fi

# check for errors
ERR_LIST=$(grep "Error" output_logs_$cur_date/xfdnn_nightly_$cur_date.csv)
echo $ERR_LIST

if [ ! -z "$ERR_LIST" ];
then
    #echo "Errors from today nightly :"
    HEADER="Errors from today's nightly :"
fi

# check for status file existence
if [ ! -f output_logs_${cur_date}/*.txt ];
then
    HEADER="Errors from today's nightly :"
    TXTFILEER_HDR="Failed to generate network status file"
fi



# check for csv file existence
if [ ! -f output_logs_${cur_date}/*.csv ];
then
    HEADER="Errors from today's nightly :"
    FILEER_HDR="Failed to generate csv file"
fi

OUT_LOG_DIR=/wrk/acceleration/test_deephi/daily_logs/output_logs_${cur_date}_Docker


if [ ! -z "$TXTFILEER_HDR" ];
then
    
    echo -e "Please find attached log for accuracy details and latency numbers. \nPlease find confluence page link for details about how to run the scripts - http://confluence.xilinx.com/display/XSW/Run+scripts+for+Deephi+networks.\n\nOutput log directory : ${OUT_LOG_DIR}\n\n\n${HEADER}\n\n${FILEER_HDR}\n${TXTFILEER_HDR}" | mailx -s "Deephi NW Accuracy using Docker - $cur_date" ${MAIL_LIST}

elif [ ! -z "$FILEER_HDR"];
then
    echo -e "Please find attached log for accuracy details and latency numbers. \nPlease find confluence page link for details about how to run the scripts - http://confluence.xilinx.com/display/XSW/Run+scripts+for+Deephi+networks.\n\nOutput log directory : ${OUT_LOG_DIR}\n\n\n${HEADER}\n\n${FILEER_HDR}\n${TXTFILEER_HDR}" | mailx -a output_logs_${cur_date}/*.txt -s "Deephi NW Accuracy using Docker - $cur_date" ${MAIL_LIST}

else
    echo -e "Please find attached log for accuracy details and latency numbers. \nPlease find confluence page link for details about how to run the scripts - http://confluence.xilinx.com/display/XSW/Run+scripts+for+Deephi+networks.\n\nOutput log directory : ${OUT_LOG_DIR}\n\n\n${HEADER}\n\n${ERR_LIST}\n\n${FILEER_HDR}\n\n${TXTFILEER_HDR}" | mailx -a output_logs_${cur_date}/*.txt -a output_logs_${cur_date}/*.csv -s "Deephi NW Accuracy using Docker - $cur_date" ${MAIL_LIST}
fi
