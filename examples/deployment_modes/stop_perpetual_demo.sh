#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
ps -u | grep "python mp_classify" | awk '{ print  $2 }' | xargs kill
