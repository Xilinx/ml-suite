#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
bash -x build_lib.sh
ssh xsjfislx18 "docker exec anup_container /bin/sh "navigate.sh""

# trigger mail
./mail.sh
