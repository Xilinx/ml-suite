###########################
## Welcome to google v1 
##
## where to exploit V3 parallelism for Conv/Pool for different byte
## per pixel -b 1 and 2 the compiler needs different
## --parallelismstrategy "*" 
## 


set -x 


DSP=96
M=9

#bin/xfdnn_compiler_caffe.py -n /wrk/hdstaff/paolod/perforce/RDI_paolod_Dev_work/src/DeepLearning/xilinx/git-version/MLsuite//models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt --dsp 96 -m 9 -s all -g googlev1.1.96.9.P --fromtensorflow --parallelread Any --pipelineconvmaxpool --poolingaround -b 1 -P --parallelismstrategy "['tops','bottom']"

FILE=${XFDNN_ROOT}/models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt
NICK=googlev1

for B in 1 
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
for B in 2 
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
 ##   python bin/xfdnn_compiler_caffe.py  -n $FILE \
 ##	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.P \
 ##	--fromtensorflow  \
 ##	--parallelread "Any" --pipelineconvmaxpool  --poolingaround -b $B \
 ##	-P --parallelismstrategy "['bottom','tops']" \
 ##	> out.${NICK}.$B.$DSP.$M.P 2> err.${NICK}.$B.$DSP.$M.P   
 ##
 ##   python bin/xfdnn_compiler_caffe.py  -n $FILE \
 ##	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.P.noreplication \
 ##	--fromtensorflow --noreplication  \
 ##	--parallelread "Any" --pipelineconvmaxpool  --poolingaround -b $B -P --parallelismstrategy "['bottom','tops']" \
 ##	> out.${NICK}.$B.$DSP.$M.P.norep 2> err.${NICK}.$B.$DSP.$M.P.norep  
 ##
 ##   python bin/xfdnn_compiler_caffe.py  -n $FILE \
 ##	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M \
 ##	--fromtensorflow  \
 ##	--parallelread "Any" --pipelineconvmaxpool  --poolingaround -b $B  --parallelismstrategy "['bottom','tops']" \
 ##	> out.${NICK}.$B.$DSP.$M 2> err.${NICK}.$B.$DSP.$M  
 ##
 ##   python bin/xfdnn_compiler_caffe.py  -n $FILE \
 ##	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M.noreplication \
 ##	--fromtensorflow  --noreplication  \
 ##	--parallelread "Any" --pipelineconvmaxpool --poolingaround  -b $B --parallelismstrategy "['bottom','tops']" \
 ##	> out.${NICK}.$B.$DSP.$M.norep 2> err.${NICK}.$B.$DSP.$M.norep   

done

B=2
M=6

for DSP in #28 56 
do
    
    python bin/xfdnn_compiler_caffe.py  -n $FILE \
	--dsp $DSP -m $M -s all -g ${NICK}.$B.$DSP.$M \
	--fromtensorflow  > out.${NICK}.$B.$DSP.$M 2> err.${NICK}.$B.$DSP.$M 



done



