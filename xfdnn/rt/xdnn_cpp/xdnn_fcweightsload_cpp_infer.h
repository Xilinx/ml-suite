
// SPDX-License-Identifier: BSD-3-CLAUSE
//
// (C) Copyright 2018, Xilinx, Inc.
//
#ifndef HDF5_CPP_INFER_H
#define HDF5_CPP_INFER_H

#include <vector>
#include <unordered_map>
void XDNNLoadFCWeights(std::unordered_map<int, std::vector<std::vector<float>> > &fc_wb_map,char *hdf5_file);

#endif
