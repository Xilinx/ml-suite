/*
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef _XHOST_H_
#define _XHOST_H_
#include "assert.h"
#include <stdio.h>
#include <vector>
#include <string>
#include <fstream>
#include "CL/cl.h"
#include "CL/cl_ext.h"
#include <boost/compute.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/program.hpp>
#include <unordered_map>
#include "gemx_util.h"

using namespace std;
namespace gemx{
typedef enum {
    OpControl, OpGemv, OpGemm, OpTransp, OpSpmv, OpResult, OpFail, OpFcn
} OpType;

class kArgs {
public:
    virtual ~kArgs() {
    }
    virtual size_t sizeInBytes() = 0;
    virtual char* asByteArray() = 0;
};

//Base address will be the instruction memory region
class XStream {
public:
    XStream() = delete;
    XStream(const boost::compute::program &p, const string & kernelName, unsigned ddrBank)
    {
        _ddrbank = ddrBank;
         m_Kernel = std::move(boost::compute::kernel(p, kernelName));
         // Create the OpenCL context to attach resources on the device
         // Create the OpenCL command queue to control the device
         //m_CommandQueue = move(boost::compute::system::default_queue());
         //boost::compute::command_queue queue(boost::compute::system::default_context(), boost::compute::system::default_device(), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
         boost::compute::command_queue queue(
                 boost::compute::system::default_context(),
                 boost::compute::system::default_device() /* CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/);
         m_CommandQueue = move(queue);
    }

    ~XStream() {
    }

    boost::compute::buffer createBuf(void *ptr, size_t sz_bytes)
    {
        cl_mem_ext_ptr_t l_bufExt;
        //l_bufExt.obj = NULL;
        l_bufExt.param = 0;
        l_bufExt.flags = _ddrbank;
        l_bufExt.obj = ptr;
        // Buffers
        return boost::compute::buffer(boost::compute::system::default_context(), sz_bytes,
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX,
                &l_bufExt);
    }

    bool copyToFpga(const boost::compute::buffer & buf, bool sync_send)
    {
        boost::compute::event l_event;
        //cout << "copyToFPGA" << endl;
        // Send the input data to the accelerator
        l_event = m_CommandQueue.enqueue_migrate_memory_objects(1, &(buf.get()),
                0);

        if (sync_send){
            l_event.wait();
        } else{
            _waitInput.insert(l_event);
        }
        return true;
    }

    boost::compute::buffer copyToFpga(void * buf, size_t sz_bytes,
            bool sync_send = false)
    {
        boost::compute::buffer cl_buf = createBuf(buf, sz_bytes);
        copyToFpga(cl_buf, sync_send);
        return cl_buf;
    }

    void copyFromFpga(const boost::compute::buffer & buf, bool sync_exec = true)
    {
        //cout << "copyFromFPGA" << endl;
        XTimer t;
        boost::compute::event l_readEvents =
                m_CommandQueue.enqueue_migrate_memory_objects(1, &(buf.get()),
                        CL_MIGRATE_MEM_OBJECT_HOST, _waitOutput);
        //l_readEvents.wait();
        if ( sync_exec ){
            l_readEvents.wait();
            _waitOutput.clear();
        } else{
            _waitOutput.insert(l_readEvents);
        }
#ifdef GEMX_PERF_DBG
        cout << "copyFromFpga: " << t.elapsed() << endl;
#endif
    }
    void execKernel(const boost::compute::buffer & instr_buf, bool sync_exec = true )
    {
        boost::compute::extents<1> offset { 0 };
        boost::compute::extents<1> global { 1 };
        // Use only 1 CU
        boost::compute::extents<1> local { 1 };
        // Launch kernels
        m_Kernel.set_args(instr_buf, instr_buf);

        XTimer t;
        //boost::compute::event l_event = m_CommandQueue.enqueue_nd_range_kernel(
        //        m_Kernel, offset, global, local, _waitInput);
        boost::compute::event l_event = m_CommandQueue.enqueue_task(m_Kernel, _waitInput);

        if ( sync_exec ) {
            l_event.wait();
        } else{
            _waitOutput.insert(l_event);
        }
        _waitInput.clear();
#ifdef GEMX_PERF_DBG
        cout << "execKernel: " << t.elapsed() << endl;
#endif

    }

    void wait ()
    {
        for (size_t i = 0; i < _waitOutput.size(); i++){
            //cout << "OpenCL event status: " <<  _waitOutput[i].status() << endl;
            _waitOutput[i].wait();
            //cout << "OpenCL event status after wait: " <<  _waitOutput[i].status() << endl;
        }
        _waitInput.clear();
        _waitOutput.clear();
    }

private:
    unsigned _ddrbank;
    boost::compute::kernel m_Kernel;
    boost::compute::context m_Context;
    boost::compute::command_queue m_CommandQueue;
    boost::compute::wait_list _waitInput, _waitOutput;
};

template<typename HType>
class XHost{
public:
    XHost() = delete;

    XHost ( const string & xclbin, const string & kernelName, unsigned ddrBank, const string &device)
    {
        //_fpga_stream.resize(nPE);
        const boost::compute::program * p = loadxclbin(xclbin);
        //vector<unsigned> ddrBanks = this->getDDRBankFlags(device);
        //cout << "Create XStream with program " << p << endl;
        _fpga_stream = shared_ptr<XStream>(new XStream(*p, kernelName, ddrBank));
    }

    virtual ~XHost(){}

    // https://gitenterprise.xilinx.com/rkeryell/heterogeneous_examples/blob/master/vector_add/SDAccel-Boost.Compute/vector_add.cpp
    // Construct an OpenCL program from the precompiled kernel file
    const boost::compute::program* loadxclbin (const string & xclbin)
    {
        static boost::compute::program p = move(
                boost::compute::program::create_with_binary_file(xclbin,
                        boost::compute::system::default_context()));
        p.build();
        return &p;
    }

    virtual void Execute( bool sync_exec = true) = 0;

    bool AddMat(const HType & handle, void * mat_ptr, unsigned long long buf_sz) {
        auto &h = _hostMat;
        auto &hz = _hostMatSz;
        if (h.find(handle) == h.end()) {
            h[handle] = mat_ptr;
            hz[handle] = buf_sz;
            return true;
        }
        else if (hz[handle] != buf_sz ){
            h[handle] = mat_ptr;
            hz[handle] = buf_sz;
            this->_devHandle.erase(handle);
            //cout << "Erasing devhandle!" << endl;
            return true;
        }
        //cout << "Matrix " << handle << " already added!" << endl;
        return false;
    }

    void * GetMat(const HType & handle,
            bool queryFPGA = false, bool sync_get = true)
    {
        auto& h = _hostMat;
        void * ret_ptr = nullptr;
        if (h.find(handle) != h.end()) {
            if (queryFPGA)
                GetFromFPGA(handle, sync_get);
            ret_ptr = h[handle];
        }
        return ret_ptr;
    }
    
    void Wait()
    {
        _fpga_stream->wait();
    }
    
    void SendToFPGA(const HType & handle, void * mat_ptr, unsigned long long buf_sz,
            bool sync_send = false) {
        AddMat(handle, mat_ptr, buf_sz);
        SendToFPGA(handle, sync_send);
    }

    void SendToFPGA(const HType & handle, bool sync_send = false) {
        XTimer t;
        auto &h = _hostMat;
        auto &d = _devHandle;
        assert(h.find(handle) != h.end());

        if (d.find(handle) != d.end()) {
            _fpga_stream->copyToFpga(d[handle], sync_send);
        } else {
            d[handle] = _fpga_stream->copyToFpga(h[handle], _hostMatSz[handle], sync_send);
        }
#ifdef GEMX_PERF_DBG
        cout << "SendToFPGA: " << t.elapsed() << endl;
#endif
    }

    void GetFromFPGA(const HType & handle, bool sync_get) {
        XTimer t;
        auto &d = _devHandle;
        assert(d.find(handle) != d.end());
        _fpga_stream->copyFromFpga(d[handle], sync_get);
#ifdef GEMX_PERF_DBG
        cout << "GetFromFPGA: " << t.elapsed() << endl;
#endif
    }

    int getBoardFreqMHz(unsigned int p_BoardId) {
      string l_freqCmd = "$XILINX_OPENCL/runtime/bin/xbsak query -d" + to_string(p_BoardId);;
      float l_freq = -1;
      char l_lineBuf[256];
      shared_ptr<FILE> l_pipe(popen(l_freqCmd.c_str(), "r"), pclose);
      if (!l_pipe) cout << ("ERROR: popen(" + l_freqCmd + ") failed");
      bool l_nextLine_isFreq = false;
      while (l_pipe && fgets(l_lineBuf, 256, l_pipe.get()) ) {
      string l_line(l_lineBuf);
      if (l_nextLine_isFreq) {
          string l_prefix, l_val, l_mhz;
          stringstream l_ss(l_line);
          l_ss >> l_prefix >> l_val >> l_mhz;
          l_freq = stof(l_val);
          assert(l_mhz == "MHz");
          break;
      } else if (l_line.find("OCL Frequency:") != string::npos) {
          l_nextLine_isFreq = true;
      }
      }
      if (l_freq == -1) {
      //if xbsak does not work, as happens on F1, put the XOCC achieved kernel frequcy here
      l_freq = -1;
      cout << "INFO: Failed to get board frequency by xbsak. This is normal for cpu and hw emulation, using -1 MHz for reporting.\n";
      }
      return((int)l_freq);
    }

protected:
    unordered_map<HType, void*  > _hostMat;
    unordered_map<HType, unsigned long long > _hostMatSz;
    unordered_map<HType, boost::compute::buffer> _devHandle;
    shared_ptr<XStream> _fpga_stream;
};


}


#endif
