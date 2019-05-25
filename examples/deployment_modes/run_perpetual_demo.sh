#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
for i in {0..0} ;
do
    unset PYTHONPATH
    ./run.sh -t streaming_classify -k v3 -b 8 -i $i -x -v > /dev/null & 
done
