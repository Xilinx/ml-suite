##############################################
## Welcome to resnet50
##
##

#set -x 


#FILE=./client-resnet/res.prototxt
#FILE=./client-resnet/goog.prototxt
FILE=$1

A="224-224 448-448 896-896 1344-1344 2016-2016 2016-4092"
#NICK=resnet50X
#NICK=googy
NICK=$2
for S in 224-224 448-448 896-896 1344-1344 2016-2016 2016-4092
do 
    DSP=96
    M=9
   
    H="$(cut -d'-' -f1 <<< $S)" 
    W="$(cut -d'-' -f2 <<< $S)"

    echo ${W}x${H}

    cat $FILE | sed -e "s/224X224_1/${H}/" | sed -e "s/224X224_2/${W}/" > $FILE.$H.$W

    for B in 1 2 
    do 
	echo "DSP=${DSP} BytePerPixel ${B} M=9 DDR=500"
	python bin/xfdnn_compiler_caffe.py  -n $FILE.$H.$W \
	    --dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.$H.$W \
	    --fromtensorflow  --ddr 500 \
	    -b $B \
	    > out.${NICK}.$B.$DSP.$M.$H.$W 2> err.${NICK}.$B.$DSP.$M.$H.$W   
	python bin/report.py --jsonfile ${NICK}.$B.$DSP.$M.$H.$W.json | grep Estimated
    done

    B=2
    M=6

    for DSP in 28 56 
    do
	echo "DSP=$DSP BytePerPixel ${B}  M=$ DDR=500"

	python bin/xfdnn_compiler_caffe.py  -n $FILE.$H.$W \
	    --dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.$H.$W  --ddr 500 \
	    --fromtensorflow  --schedulefile resnet.txt > out.${NICK}.$B.$DSP.$M 2> err.${NICK}.$B.$DSP.$M 
	python bin/report.py --jsonfile ${NICK}.$B.$DSP.$M.$H.$W.json | grep Estimated  
    done
    
done 
