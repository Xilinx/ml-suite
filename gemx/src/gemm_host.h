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
#ifndef _GEMM_HOST_H
#define _GEMM_HOST_H

#include "xhost.h"
using namespace std;

namespace gemx{

class GemmArgs: public kArgs {
public:
    virtual ~GemmArgs() {
    }
    GemmArgs() = delete;
    GemmArgs(unsigned int p_Aoffset, unsigned int p_Boffset,
            unsigned int p_Coffset, unsigned int p_Xoffset, unsigned int p_M, unsigned int p_K,
            unsigned int p_N, unsigned int p_Lda, unsigned int p_Ldb,
            unsigned int p_Ldc, unsigned int p_Ldx, int post_scale, int post_shift) :
                m_gemm_args( { int(OpGemm),  p_Aoffset, p_Boffset, p_Coffset, p_Xoffset, p_M, p_K,
        p_N, p_Lda, p_Ldb, p_Ldc, p_Ldx, 0, 0, 0, 0 }) {
        m_gemm_args.m_postScaleVal = (post_scale << 8) | (post_shift & 0x000000ff);
    }
    size_t sizeInBytes() {
        return sizeof(m_gemm_args);
    }
    char *asByteArray() {
        return reinterpret_cast<char*>(&m_gemm_args);
    }

protected:
    struct {
        int m_optype;
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_Xoffset, m_M, m_K, m_N,
        m_Lda, m_Ldb, m_Ldc, m_Ldx;
    int m_postScaleVal;
        int dummy[3];
    } m_gemm_args;
};


template<typename HType>
class GEMMHost : public XHost<HType> {
public:
    GEMMHost() = delete;

    virtual ~GEMMHost() {}

    GEMMHost(const GEMMHost<HType> &) = delete;


    static vector<unsigned> getDDRBankFlags(const string & device)
    {
        vector<unsigned>ddrBanks;
        if ( device == "ku115" || device == "kcu1500" ){
            ddrBanks = {XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK1, XCL_MEM_DDR_BANK2, XCL_MEM_DDR_BANK3};
        }
        else if( device == "vcu1525"){
            ddrBanks = {XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK3, XCL_MEM_DDR_BANK1, XCL_MEM_DDR_BANK2 };
        }
        else if ( device == "vu9pf1"){
            ddrBanks = {XCL_MEM_DDR_BANK0, XCL_MEM_DDR_BANK2, XCL_MEM_DDR_BANK3, XCL_MEM_DDR_BANK1};
        }
        else{
            cerr << "Unsupported device! Options are ku115, kcu1500, vu9pf1, vcu1525" << endl;
            assert( device == "ku115" || device == "kcu1500" || device == "vu9pf1" || device == "vcu1525");
        }
        return ddrBanks;
    }

    static string getKernelName(unsigned PE)
    {
    	return "gemxKernel_" + std::to_string(PE);
    }

    GEMMHost(const string & xclbin, const string & kernelName, unsigned ddrBank, const string &device) : XHost<HType> ( xclbin, kernelName, ddrBank, device)
    {
        void *aligned_mem = nullptr;
        assert(!posix_memalign(&aligned_mem, PAGE_SIZE, INSTR_BUF_SIZE));
        _instrBuf = shared_ptr<char>((char*) aligned_mem);
        memset(_instrBuf.get(), 0, INSTR_BUF_SIZE);
        _instr_offset = 0;
        this->_cl_instr_buf = this->_fpga_stream->copyToFpga(_instrBuf.get(), INSTR_BUF_SIZE,
                true);
        xclGetMemObjDeviceAddress(this->_cl_instr_buf.get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &this->_ddrDeviceBaseAddr);

        assert(!posix_memalign(&aligned_mem, PAGE_SIZE, KERN_DBG_BUF_SIZE));
        _kernDbgBuf = shared_ptr<char>((char*) aligned_mem);
        _cl_kern_dbg_buf = this->_fpga_stream->copyToFpga(_kernDbgBuf.get(),
                KERN_DBG_BUF_SIZE, true);
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift) {
        return AddGEMMOp (A, B, C, bias, m, k, n, k, n, n, n, postScale, postShift);
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift) {
        XTimer t;
        if (this->_hostMat.find(A) == this->_hostMat.end()
                || this->_hostMat.find(B) == this->_hostMat.end()
                || this->_hostMat.find(C) == this->_hostMat.end()
                || this->_hostMat.find(bias) == this->_hostMat.end()) {
            cerr << "Matrix not found!" << endl;
            return false;
        }
        unsigned long long A_off = 0, B_off = 0, C_off = 0, X_off = 0;

        xclGetMemObjDeviceAddress(this->_devHandle[A].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
        xclGetMemObjDeviceAddress(this->_devHandle[B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
        xclGetMemObjDeviceAddress(this->_devHandle[C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);

        if ( this->_devHandle.find(bias) != this->_devHandle.end()){
            xclGetMemObjDeviceAddress(this->_devHandle[bias].get(),
                    boost::compute::system::default_device().get(),
                    sizeof(unsigned long long), &X_off);
            assert(X_off > this->_ddrDeviceBaseAddr);
            X_off -= this->_ddrDeviceBaseAddr;
        }

       // cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << " X_dev_addr: " << X_off << endl;
        assert(A_off > this->_ddrDeviceBaseAddr);
        assert(B_off > this->_ddrDeviceBaseAddr);
        assert(C_off > this->_ddrDeviceBaseAddr);
        A_off -= this->_ddrDeviceBaseAddr;
        B_off -= this->_ddrDeviceBaseAddr;
        C_off -= this->_ddrDeviceBaseAddr;

        assert(A_off % PAGE_SIZE == 0);
        assert(B_off % PAGE_SIZE == 0);
        assert(C_off % PAGE_SIZE == 0);
        assert(X_off % PAGE_SIZE == 0);

        A_off /= PAGE_SIZE;
        B_off /= PAGE_SIZE;
        C_off /= PAGE_SIZE;
        X_off /= PAGE_SIZE;

        GemmArgs gargs(A_off, B_off, C_off, X_off, m,
                k, n, lda, ldb, ldc, ldx, postScale, postShift);
        this->AddInstr ( &gargs);
        return true;
    }

    virtual void Execute( bool sync_exec = true) {
        XTimer t;
        this->_fpga_stream->copyToFpga(this->_cl_instr_buf, false);
        this->_fpga_stream->execKernel(this->_cl_instr_buf, sync_exec);
#ifdef GEMX_PERF_DBG
        cout << "Execute: " << t.elapsed() << endl;
#endif
    }

    virtual void ClearInstrBuf()
    {
        memset(this->_instrBuf.get(), 0, PAGE_SIZE);
        this->_instr_offset = 0;
    }
protected:

    void AddInstr  ( kArgs * args )
    {
        char * instr = args->asByteArray();
        char * curr_pos = &_instrBuf.get()[_instr_offset];
        memcpy(curr_pos, instr, args->sizeInBytes());
        _instr_offset += args->sizeInBytes();
    }
    static const unsigned int PAGE_SIZE = 4096;
    static const unsigned int INSTR_BUF_SIZE = PAGE_SIZE;
    static const unsigned int KERN_DBG_BUF_SIZE = PAGE_SIZE;

    unsigned long long _ddrDeviceBaseAddr;
    shared_ptr<char> _instrBuf, _kernDbgBuf;
    boost::compute::buffer _cl_instr_buf, _cl_kern_dbg_buf;
    unsigned int _instr_offset;
};

}
#endif
