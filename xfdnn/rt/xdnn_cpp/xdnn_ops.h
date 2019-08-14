// SPDX-License-Identifier: BSD-3-CLAUSE
//
// (C) Copyright 2018, Xilinx, Inc.
//
#ifndef XDNN_OPS_H
#define XDNN_OPS_H

#include <vector>
#include <unordered_map>
#include <CL/opencl.h>
#include <boost/thread.hpp>
#define XDNN_ONEHACK 0

// using RegisterWriteManager from xdnn namespace
using xdnn::RegisterWriteManager;

class XBLASHandle;
class XPipelinePacket;
class XComputeUnit;
/****************************************
 * KernelInterface (CSR reader/writer)
 ****************************************/
int XDNNCrossBar( XComputeUnit * cu, std::shared_ptr <XDNNDescriptorDB> descDB, const std::string &compilerJson );

class KernelInterface {
public:
  KernelInterface() = delete;
  KernelInterface(XBLASHandle *handle);

  long computeCsrBaseAddr(int cuIdx, int blockIdx);
  void writeRegister(int cuIdx, int blockIdx, size_t offset, size_t size, const void* data, RegisterWriteManager *rwm = nullptr);
  void writeRegisterDebug();
  void readHardwareCounter();

  int readImageExecutionCounter(const int cuIdx, const int slice);
  void extractKernelConfig();
  void readRegister(int cuIdx, size_t offset, size_t size, void* data);

  void writeInt(XComputeUnit * cu, size_t offset, int value);
  void writeInt(int cuIdx, int blockIdx, size_t offset, int value, RegisterWriteManager *rwm = nullptr);
  void writeInt(int cuIdx, size_t offset, int value, RegisterWriteManager *rwm = nullptr);
  void writeIntAllBlocks(int cuIdx, size_t offset, int value, RegisterWriteManager *rwm = nullptr);

  void writeImgDdrBase(int cuIdx, long long value, RegisterWriteManager *rwm = nullptr);

  void writeFilterDdrBase(int cuIdx, long long value, RegisterWriteManager *rwm = nullptr);

  void enableXRT(XComputeUnit * cu, bool isThroughput);

  void pushCmd(size_t dl_ofst, size_t ul_ofst, XPipelinePacket *packet);

  size_t getInstrCounter(size_t cuidx)
  {
      return _instrCounter[cuidx];
  }
  size_t getInstrCounter(XComputeUnit *cu) {
      return getInstrCounter ( cu->getCuIdx());
  }

  size_t getDFLInstrCounter(size_t cuidx)
  {
      return _dflInstrCounter[cuidx];
  }
  size_t getDFLInstrCounter(XComputeUnit *cu) {
      return getDFLInstrCounter ( cu->getCuIdx());
  }

private:
  XBLASHandle *_handle;
  cl_mem _regMap; // "handle" to this CU's CSR region
  std::vector<std::vector<size_t> > _csrBase;
  std::vector<int> _cuIdx2SlrIdx;
  std::vector<bool> _wroteImgDdrBase;
  std::unordered_map<size_t, size_t> _instrCounter;
  std::unordered_map<size_t, size_t> _dflInstrCounter;

};

class KernelInterfaceDB{
public:
    KernelInterfaceDB() = delete;
    KernelInterfaceDB(KernelInterfaceDB const&) = delete;             // Copy construct
    KernelInterfaceDB(KernelInterfaceDB&&) = delete;                  // Move construct
    KernelInterfaceDB& operator=(KernelInterfaceDB const&) = delete;  // Copy assign
    KernelInterfaceDB& operator=(KernelInterfaceDB &&) = delete;      // Move assign

    static KernelInterface* get(XBLASHandle *handle)
    {
        static std::map<int, std::shared_ptr<KernelInterface>> g_kis;
        static boost::mutex m;
        boost::lock_guard < boost::mutex > guard(m);
        if (g_kis.find(handle->getId()) == g_kis.end())
            g_kis[handle->getId()] = std::shared_ptr<KernelInterface>( new KernelInterface(handle));

        return g_kis[handle->getId()].get();
    }

};


#endif
