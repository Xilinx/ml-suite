##############################################
## Welcome to resnet50
##
##

set -x 

DSP=96
M=9

FILE=${XFDNN_ROOT}/models/caffe/resnet/fp32/resnet50_without_bn_deploy.prototxt
NICK=resnet50

for B in 1 2 
do
    
    python bin/xfdnn_compiler_caffe.py  -n $FILE \
	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.simple \
	--fromtensorflow  \
	-b $B \
	--noreplication  \
	> out.${NICK}.$B.$DSP.$M.P 2> err.${NICK}.$B.$DSP.$M.P   

    python bin/xfdnn_compiler_caffe.py  -n $FILE \
	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.replication \
	--fromtensorflow  \
	-b $B \
	 \
	> out.${NICK}.$B.$DSP.$M.P 2> err.${NICK}.$B.$DSP.$M.P   

##    python bin/xfdnn_compiler_caffe.py  -n $FILE \
##	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.P \
##	--fromtensorflow  \
##	--parallelread "Any" --pipelineconvmaxpool  --poolingaround -b $B \
##	-P --parallelismstrategy "['tops','bottom']" \
##	> out.${NICK}.$B.$DSP.$M.P 2> err.${NICK}.$B.$DSP.$M.P   
##
##    python bin/xfdnn_compiler_caffe.py  -n $FILE \
##	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.P.noreplication \
##	--fromtensorflow --noreplication  \
##	--parallelread "Any" --pipelineconvmaxpool  --poolingaround -b $B -P --parallelismstrategy "['tops','bottom']" \
##	> out.${NICK}.$B.$DSP.$M.P.norep 2> err.${NICK}.$B.$DSP.$M.P.norep 
##
##    python bin/xfdnn_compiler_caffe.py  -n $FILE \
##	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M \
##	--fromtensorflow  \
##	--parallelread "Any" --pipelineconvmaxpool  --poolingaround -b $B  --parallelismstrategy "['tops','bottom']" \
##	> out.${NICK}.$B.$DSP.$M 2> err.${NICK}.$B.$DSP.$M  
##
##    python bin/xfdnn_compiler_caffe.py  -n $FILE \
##	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.noreplication \
##	--fromtensorflow  --noreplication  \
##	--parallelread "Any" --pipelineconvmaxpool --poolingaround  -b $B --parallelismstrategy "['tops','bottom']" \
##	> out.${NICK}.$B.$DSP.$M.norep 2> err.${NICK}.$B.$DSP.$M.norep   

done


B=2
M=4

for DSP in #28 56 
do
    
    python bin/xfdnn_compiler_caffe.py  -n $FILE \
	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M \
	--fromtensorflow   > out.${NICK}.$B.$DSP.$M 2> err.${NICK}.$B.$DSP.$M 

done



