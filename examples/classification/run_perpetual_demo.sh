for i in {0..0} ;
do
    unset PYTHONPATH
    ./run.sh -t streaming_classify -k v3 -b 8 -c throughput -i $i -x -v > /dev/null & 
done
