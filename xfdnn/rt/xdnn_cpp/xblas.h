// SPDX-License-Identifier: BSD-3-CLAUSE
//
// (C) Copyright 2018, Xilinx, Inc.
//
/**
 *  @brief Xilinx XFDNN library for FPGA acceleration
 *
 *  @author Aaron Ng (aaronn@xilinx.com), Bill Teng (xteng@xilinx.com)
 */

#ifndef XBLAS_H
#define XBLAS_H

#include <chrono>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <CL/opencl.h>
#include <boost/thread.hpp>
#include <boost/align/aligned_allocator.hpp>
#include <fstream>
#include <memory>
#include "xdnn_util.h"
#include "thread_pool.h"
struct xrt_device;
#define XDNN_ONEHACK 0
#define XDNN_USE_JOB_PARALLELISM 0
#define XDNN_USE_JOB_THREADS 1
#define XDNN_USE_JOB_THREADS_TOTAL_THREADS 16
#define XDNN_USE_JOB_THREADS_MAX_STREAMS 32

#if defined(ZMQ)
#include <zmq.hpp>
#include <zmq_utils.h>
#endif

namespace std {
namespace {
// [aaronn] hash function for tuples
// Code from boost
// Reciprocal of the golden ratio helps spread entropy
//     and handles duplicates.
// See Mike Seymour in magic-numbers-in-boosthash-combine:
//     http://stackoverflow.com/questions/4948780

template<class T>
inline void hash_combine(std::size_t& seed, T const& v) {
  seed ^= hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Recursive template code derived from Matthieu M.
template<class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
struct HashValueImpl {
  static void apply(size_t& seed, Tuple const& tuple) {
    HashValueImpl<Tuple, Index - 1>::apply(seed, tuple);
    hash_combine(seed, get<Index>(tuple));
  }
};

template<class Tuple>
struct HashValueImpl<Tuple, 0> {
  static void apply(size_t& seed, Tuple const& tuple) {
    hash_combine(seed, get<0>(tuple));
  }
};
}

template<typename ... TT>
struct hash<std::tuple<TT...>> {
  size_t operator()(std::tuple<TT...> const& tt) const {
    size_t seed = 0;
    HashValueImpl<std::tuple<TT...> >::apply(seed, tt);
    return seed;
  }
};
}

class XMemPtr;

enum XBLASKernelType {
  XDNN_KERNEL
};
class XBLASKernelConfig {
public:
  XBLASKernelConfig();

  int m_numKernels;
  int m_forceRunOnKernelIdx;
  XBLASKernelType m_kernelType;

  // DDR mappings for each kernel & cu
  std::vector<std::vector<int> > m_ddrBankA;
  std::vector<std::vector<int> > m_ddrBankB;
  std::vector<std::vector<int> > m_ddrBankC;

  std::vector<int> m_configRegInfo;
  std::map<std::string, std::string> m_xclbinJsonVals;

  void setDDRMap(const std::string &str);
  int getDDRMap(int i);

  void print();

private:
  std::vector<int> m_ddrBankMap;
};

class XBLASConfig {
public:
  XBLASConfig();
  void print();

  bool m_useCblas;
  bool m_async; // if async, user must call waitForResults()
  std::string m_taskName;
};

class XTimer {
public:
  XTimer() :
      beg_(clock_::now()) {
  }
  void reset() {
    beg_ = clock_::now();
  }
  double elapsed() const {
    return std::chrono::duration_cast < second_ > (clock_::now() - beg_).count();
  }

private:
  typedef std::chrono::high_resolution_clock clock_;
  typedef std::chrono::duration<double, std::ratio<1> > second_;
  std::chrono::time_point<clock_> beg_;
};

class XComputeUnitManager;
class XBLASWorkers {
public:
  static XBLASWorkers &get() {
    static XBLASWorkers impl_;
    return impl_;
  }

  XBLASWorkers() :
      m_threadsAlive(false), m_numPrepThreads(1), m_numExecThreads(1), m_numPostThreads(
          1) {
  }

  ~XBLASWorkers(){
  }


  // thread management
  void setupThreads();
  void destroyThreads();

  bool m_threadsAlive;
  int m_numPrepThreads;
  int m_numExecThreads;
  int m_numPostThreads;
  boost::mutex m_mtx;
};
class XPipelinePacket;
class XBLASHandle {
public:
  static XBLASHandle* get(int idx = 0, bool create = true);
  static int getNum() {
    return XBLASHandle::getInstance().size();
  }

  // should not be public
  // but this is ONEHACK life
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_device_id device_id;
  std::vector<cl_kernel> kernels;
  xrt_device* xdev;
#if defined(ZMQ)
  zmq::context_t *g_zmqcontext;
  zmq::socket_t *g_zmqsocket;
#endif

  // yuck, need to clean up XBLAShandle 
  // don't expose guts like this
  std::vector<XComputeUnitManager*> m_computeUnitManagers;

  bool m_isInitialized;
  XBLASKernelConfig m_kConfig;
  XBLASConfig m_config;

  int getId() {
    return m_id;
  }

  void processXclbinJson(const std::string &path);

  // m_host2XMemMap helpers
  void setHost2XMem(const void *src, XMemPtr* dest);
  XMemPtr* getHost2XMem(const void *src);
  void deleteHost2XMem(const void *src);
  void destroyDeviceMemPODs();

  void waitForResults(int streamId = 0, bool recordPipelineTimes = false);
  int queryStream(int streamId = 0);

  // multi-handle management
  void addRefCount();
  int delRefCount();
  void release(int idx = 0);

  // JobParrallelism
  void JobParrallelism(XPipelinePacket *packetPtr);

private:
  XBLASHandle(int id = 0); // handled by get()
  ~XBLASHandle(); // handled by release()



  static std::vector<XBLASHandle*>& getInstance(); // singleton

  // store a mapping of all xMemcpy (src, dest) ptrs
  // this is so we can map src ptrs to XMemPtrs.
  std::map<const void*, XMemPtr*> m_host2XMemMap;

  // make one command queue for each thread
  std::map<unsigned long, cl_command_queue> m_commands;

  int m_id;
  boost::mutex m_timer_mtx;
  boost::mutex m_mtx;
  int m_refCnt;
  //JobParrallelism
  ThreadPool *m_workerPool;
  // first dimension the stream
  // second dimension the job (one job per image, size is batch size)
  std::vector<std::vector<std::future<void>> > m_workerPoolResult;
};


template <typename DType>
class XDeviceMemPOD {
public:
  XDeviceMemPOD(): _hostsz(0), _hostptr(nullptr), m_devicePtr(nullptr){}
  ~XDeviceMemPOD(){}
  size_t numelem() { return _hostsz; }
  //size_t deviceSizeInBytes() { return sizeof(DType) * _deviceSz; }
  xdnn::XAlignedBlob<DType> & getData() { return m_data; }

  size_t _hostsz;
  void * _hostptr;
  xdnn::XAlignedBlob<DType> m_data; // (optional) can be used to store transformed data
                             //            (e.g., for padding)
  cl_mem m_devicePtr;
};

// Class to manage host<->device memory alloc and transfer
// Note: 
// - XMemPtr maps 1-to-1 to hostPtr
// - in multi-kernel/multi-cu mode, 
//   XMemPtr maps 1 hostPtr to many XDeviceMemPOD<short> objects
class XMemPtr {
public:
  XMemPtr(void * ptr, size_t sz);
  ~XMemPtr();

  std::pair<int, int> getKernelComputeUnitLocation(const void*);
  XDeviceMemPOD<short>* getDeviceMemPOD(int kIdx, int cuIdx);
  void deleteDeviceMemPODs();
  size_t size() const { return m_sizeInBytes; }
  void * get() const { return m_srcPtr; }
private:
  // populated by use r
  size_t m_sizeInBytes;
  void *m_srcPtr; // pointer to original data, in case we need lookup
  // auto-populated when data is written from host to device
  // this holds the data objects prepped for the FPGA
  // m_hostDeviceMemMap helpers
  // Note: 
  // - each kernel's CU can have its own DDRs
  // - one host ptr can map to multiple CUs' DDRs
  //   (e.g. A blob is loaded into all CUs' DDRs) 
  std::unordered_map<std::tuple<const void*, int, int>, XDeviceMemPOD<short> * > m_hostDeviceMemMap;
  XMemPtr();
};

char* allocateScratchSpace(XBLASHandle &handle, int bytes, int prefCuIdx);
int xdnnv3ImgSizeReq(int, int, int, int, int, int);
template<class DestT>
void std2xdnnv3(short*, DestT*, int, int, int, int, int, int);
template<class DdrT>
void xdnnv32std(DdrT*, short*, int, int, int, int, int, int);
/******************************************************************* 
 * XBLAS Client API
 *******************************************************************/

class XPipelinePacket;
void xblasEnqueueJob(XBLASHandle &handle, XPipelinePacket *pkt);

template<typename DType>
void XDNNQuantize(const float *in, DType *out, const int sz, const float thresh, const int bitWidth, const unsigned out_stride = 1, bool doRounding=false);

extern "C" {
int xblasCreate(XBLASHandle *&handle, const char *xclbin,
    const char *kernelName, int deviceIdx = 0);

void xFree(XBLASHandle &handle, XMemPtr *memPtr, bool freeMemPtr = true);

XMemPtr* xRegMem(XBLASHandle &handle, void *hostMemPtr, size_t sz, bool copy = true);

void xblasDestroy(XBLASHandle *handle);

// special functions to preload matrices to DDR
void xblasLoadA(XBLASHandle &handle,  short *A, const char *layerMapStr, int prefCuIdx = -1);

#ifdef ENABLE_HBM
void xblasHBMLoadA(XBLASHandle &handle,  std::vector<std::vector<int> > &dataBlobsHBM, const char *layerMapStr, int prefCuIdx = -1);
#endif

void computeFC(float *weight, float *bias, float *data, int M, int N, int K,
    float *output);
void computeSoftmax(float * input, size_t bs, size_t num_elem);

} // extern "C" {

#endif /*XBLAS_H*/
