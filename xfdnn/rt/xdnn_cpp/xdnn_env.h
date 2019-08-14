// SPDX-License-Identifier: BSD-3-CLAUSE
//
// (C) Copyright 2018, Xilinx, Inc.
//
#ifndef XDNN_ENV_H
#define XDNN_ENV_H

#include <sstream>

const float XLNX_GLOBAL_SCALE_A =
    std::getenv("XDNN_GLOBAL_SCALE_A") ?
        atoi(std::getenv("XDNN_GLOBAL_SCALE_A")) : -1;
const float XLNX_GLOBAL_SCALE_B =
    std::getenv("XDNN_GLOBAL_SCALE_B") ?
        atoi(std::getenv("XDNN_GLOBAL_SCALE_B")) : -1;
const std::string XLNX_KERNEL_XCLBIN =
    std::getenv("XDNN_KERNEL_XCLBIN") ?
        std::string(std::getenv("XDNN_KERNEL_XCLBIN")) : "kernelSgemm.xclbin";
const std::string XLNX_KERNEL_BASE_NAME =
    std::getenv("XDNN_KERNEL_BASE_NAME") ?
        std::string(std::getenv("XDNN_KERNEL_BASE_NAME")) : "kernelSxdnn_0";
const int XBLAS_NUM_DEVICES =
    std::getenv("XBLAS_NUM_DEVICES") ?
        atoi(std::getenv("XBLAS_NUM_DEVICES")) : 1;

const bool XDNN_VERBOSE =
    std::getenv("XDNN_VERBOSE") ?
        atoi(std::getenv("XDNN_VERBOSE")) == 1 : false;
const bool XLNX_DUMP_XDNN_STANDALONE_DATA =
    std::getenv("XDNN_DUMP_STANDALONE_DATA") ?
        atoi(std::getenv("XDNN_DUMP_STANDALONE_DATA")) == 1 : false;

const bool XDNN_EMIT_PROFILING_INFO =
  std::getenv("XDNN_EMIT_PROFILING_INFO") ?
    atoi(std::getenv("XDNN_EMIT_PROFILING_INFO")) == 1 : false;
const bool XBLAS_EMIT_PROFILING_INFO = 
  (XDNN_EMIT_PROFILING_INFO)? XDNN_EMIT_PROFILING_INFO 
  : (std::getenv("XBLAS_EMIT_PROFILING_INFO") ?
       atoi(std::getenv("XBLAS_EMIT_PROFILING_INFO")) == 1 : false);

const bool XDNN_READ_HARDWARE_GENERAL_COUNTER = 
    std::getenv("XDNN_READ_HARDWARE_GENERAL_COUNTER") ?
      atoi(std::getenv("XDNN_READ_HARDWARE_GENERAL_COUNTER")) == 1 : false;
const bool XDNN_HARDWARE_GENERAL_COUNTER_TIME_DL =
    std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_DL") ?
    atoi(std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_DL")) == 1 : false;
const bool XDNN_HARDWARE_GENERAL_COUNTER_TIME_UL =
    std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_UL") ?
    atoi(std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_UL")) == 1 : false;
const bool XDNN_HARDWARE_GENERAL_COUNTER_TIME_FL =
    std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_FL") ?
    atoi(std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_FL")) == 1 : false;
const bool XDNN_HARDWARE_GENERAL_COUNTER_TIME_MISC =
    std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_MISC") ?
    atoi(std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_MISC")) == 1 : false;
const bool XDNN_HARDWARE_GENERAL_COUNTER_TIME_SYSARR =
    std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_SYSARR") ?
    atoi(std::getenv("XDNN_HARDWARE_GENERAL_COUNTER_TIME_SYSARR")) == 1 : false;

// ONE HACK life
const bool XDNN_ONE_HACK_MASTER =
  std::getenv("XDNN_ONE_HACK_MASTER") ?
    atoi(std::getenv("XDNN_ONE_HACK_MASTER")) == 1 : false;
const bool XDNN_ONE_HACK_SLAVE =
  std::getenv("XDNN_ONE_HACK_SLAVE") ?
    atoi(std::getenv("XDNN_ONE_HACK_SLAVE")) == 1 : false;
const long long XDNN_ONE_HACK_IMG_DDR_BASE =
  std::getenv("XDNN_ONE_HACK_IMG_DDR_BASE") ?
    std::stoll(std::getenv("XDNN_ONE_HACK_IMG_DDR_BASE"), NULL, 16) : -1;
const long long XDNN_ONE_HACK_FILTER_DDR_BASE =
  std::getenv("XDNN_ONE_HACK_FILTER_DDR_BASE") ?
    std::stoll(std::getenv("XDNN_ONE_HACK_FILTER_DDR_BASE"), NULL, 16) : -1;

namespace xdnn {

inline std::string GetXclbin() {
  return XLNX_KERNEL_XCLBIN;
}

inline std::string GetKernelBaseName() {
  return XLNX_KERNEL_BASE_NAME;
}

}

#endif
