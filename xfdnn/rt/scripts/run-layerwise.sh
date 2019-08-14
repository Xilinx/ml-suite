usage() {
    echo -e "\nUsage: ./run-layerwise.sh --platform <platform> -cn <compiler json> -cq <quantizer json> -cw <.h5 file>\n"
}
# Parse Options
OPTS=`getopt -o p:t:m:k:b:d:s:a:n:ns:i:c:y:gvzfxh --long platform:,test:,model:,kcfg:,bitwidth:,directory:,numdevices:,numstreams:,deviceid:,batchsize:,compilerOpt:,numprepproc,checkaccuracy,verbose,zelda,frodo,perpetual,help -n "$0" -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi

while true
do
    case "$1" in
        -p |--platform      ) MLSUITE_PLATFORM="$2" ; shift 2 ;;
        -cn|--customnet     ) CUSTOM_NETCFG="$2"    ; shift 2 ;;
        -cq|--customquant   ) CUSTOM_QUANTCFG="$2"  ; shift 2 ;;
        -cw|--customwts     ) CUSTOM_WEIGHTS="$2"   ; shift 2 ;;
        -h |--help          ) usage                 ; exit  1 ;;
        *) break ;;
    esac
done
# Set Platform Environment Variables
if [ -z $MLSUITE_ROOT ]; then
    MLSUITE_ROOT=../..
fi
. ${MLSUITE_ROOT}/overlaybins/setup.sh ${MLSUITE_PLATFORM}

TEST=layerwise.py
BATCHSIZE=1

if [ ! -z $CUSTOM_NETCFG ]; then
    NETCFG=$CUSTOM_NETCFG
else
    echo "[ERROR] Expected Compiler JSON : No such file" ; usage
    exit 1
fi
if [ ! -z $CUSTOM_WEIGHTS ]; then
    WEIGHTS=$CUSTOM_WEIGHTS
else
    echo "[ERROR] Expected Weights : No such file" ; usage 
    exit 1
fi
if [ ! -z $CUSTOM_QUANTCFG ]; then
    QUANTCFG=$CUSTOM_QUANTCFG
else 
    echo "[ERROR] Expected Quantizer JSON : No such file" ;  usage
    exit 1
fi

XCLBIN=overlay_4.xclbin
index=0
previousLatency=0.0
totalLatency=0.0
BASEOPT="--xclbin ${XCLBIN_PATH}/${XCLBIN}
         --netcfg $NETCFG
         --weights $WEIGHTS
         --quantizecfg $QUANTCFG
         --batch_sz $BATCHSIZE
         --images ${MLSUITE_ROOT}/examples/deployment_modes/dog.jpg"
#echo $BASEOPT
printf '%s\n' "--------------------------------------------------"
printf '%-30s   %.8s\n' "Layer Name" "Time"
printf '%s\n' "--------------------------------------------------"
# Run each layer, one by one
OIFS=$IFS
while :
do
    layertime=`python ${TEST} --layerindex ${index} ${BASEOPT} | grep " = "`
    index=$((index+1))
    IFS=" = "
    read -ra Arr <<< "$layertime"
    layname=${Arr[0]}
    latency=${Arr[1]}
    #echo $layname $latency
    IFS=$OIFS
    if test "$latency" = "Done"; then
        break 
    elif test "$latency" = "0"; then
        printf '%-30s = %s\n' "$layname" "$latency"
        continue
    fi
    timetaken=`awk '{print $1-$2}' <<< "$latency $previousLatency"`
    totalLatency=`awk '{print $1+$2}' <<< "$totalLatency $timetaken"`
    previousLatency=$latency
    printf '%-30s = %s\n' "${layname}" "${timetaken}"
done

printf '%s\n' "--------------------------------------------------"
printf '%-30s = %.8s\n' "Total Time" "${totalLatency}"
printf '%s\n' "--------------------------------------------------"
`rm *.json`

IFS=$OIFS
