#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
ssh 10.22.64.112 "cd /wrk/acceleration/MLsuite_Embedded/anup/deephi_quant/MLsuite/test_deephi && hostname && bash -x nw_list.sh"
#nw_list.sh"

# trigger mail
./mail.sh
