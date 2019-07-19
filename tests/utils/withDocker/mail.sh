
cur_date=$(date +"%d%b%y")

echo -e "Please find attached log for accuracy details and latency numbers. \nPlease find confluence page link for details about how to run the scripts - http://confluence.xilinx.com/display/XSW/Run+scripts+for+Deephi+networks" | mailx -a output_logs/*.txt -a output_logs/*.csv -s "Deephi NW Accuracy using Docker - $cur_date" asirasa@xilinx.com aaronn-all@xilinx.com elliott-all@xilinx.com sumitn-all@xilinx.com
#echo -e "Please find attached log for accuracy details and latency numbers. \nPlease find confluence page link for details about how to run the scripts - http://confluence.xilinx.com/display/XSW/Run+scripts+for+Deephi+networks" | mailx -a output_logs/*.txt -a output_logs/*.csv -s "Deephi NW Accuracy using docker - $cur_date" anup@xilinx.com 
#asirasa@xilinx.com aaronn-all@xilinx.com elliott-all@xilinx.com sumitn-all@xilinx.com
