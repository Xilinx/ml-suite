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
#ifndef _FCN_HOST_H_
#define _FCN_HOST_H_

#include "gemm_host.h"
#include "gemx_util.h"
#include "xhost.h"
using namespace std;

namespace gemx{

class FcnArgs: public kArgs {
public:
    virtual ~FcnArgs() {
    }
    FcnArgs() = delete;
    FcnArgs(unsigned int p_Aoffset, unsigned int p_Boffset,
            unsigned int p_Coffset, unsigned int p_Xoffset, unsigned int p_M, unsigned int p_K,
            unsigned int p_N, unsigned int p_Lda, unsigned int p_Ldb,
            unsigned int p_Ldc, unsigned int p_Ldx, int post_scale, int post_shift, short prelu_scale, short prelu_alpha) :
                m_fcn_args( { OpFcn, p_Aoffset, p_Boffset, p_Coffset, p_Xoffset, p_M, p_K,
        p_N, p_Lda, p_Ldb, p_Ldc, p_Ldx, 0, 0, 0, 0 }) {

        m_fcn_args.m_postScaleVal = (post_scale << 8) | (post_shift & 0x000000ff);
        m_fcn_args.m_PReLUVal = (prelu_scale << 6) | (prelu_alpha & 0x003f);

        //cout << "s_dummy: " << m_fcn_args.s_dummy << endl;
        //printf ("PReLUVal: %d\n", m_fcn_args.m_PReLUVal);
        //stringstream stream;
        //cout << "optype: " << optype << " p_Aoffset: " << p_Aoffset << endl;

        /*
         int * data = (int*)asByteArray();
         for (int i = 0; i < sizeInBytes()/4; i++){
             cout << "word " << i << ": " << data[i] << endl;
         }
         */

        //string result( stream.str() );
        //cout << "Hex: " << result << endl;
    }
    size_t sizeInBytes() {
        return sizeof(m_fcn_args);
    }
    char *asByteArray() {
        return reinterpret_cast<char*>(&m_fcn_args);
    }

protected:
    struct {
        int m_optype;
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_Xoffset, m_M, m_K, m_N,
        m_Lda, m_Ldb, m_Ldc, m_Ldx;
        int m_postScaleVal;
        short m_PReLUVal;
        short s_dummy;
        int dummy[2];
    } m_fcn_args;
};
template<typename HType>
class FCNHost : public GEMMHost <HType>
{
public:
    FCNHost() = delete;
    virtual ~FCNHost(){}
    FCNHost ( const FCNHost<HType>&) = delete;
    FCNHost(const string & xclbin, const string & kernelName, const unsigned ddrBank, const string & device) : GEMMHost<HType> ( xclbin, kernelName, ddrBank, device)
    {
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift)
    {
        return AddFCNOp (A, B, C, bias, m, k, n, k, n, n, n, postScale, postShift, 1, 0);
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift) {
        return AddFCNOp (A, B, C, bias, m, k, n, k, n, n, n, postScale, postShift, 1, 0);
    }

    virtual bool AddFCNOp ( const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha)
    {
        return AddFCNOp ( A, B, C, bias, m, k, n, k, n, n, n,postScale, postShift, PReLUScale, PReLUAlpha);
    }

    virtual bool AddFCNOp ( const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift, short PReLUScale, short PReLUAlpha)
    {
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

        //cout << "A_dev_addr: " << A_off << " B_dev_addr: " << B_off << " C_dev_addr: " << C_off << endl;
        assert(A_off > this->_ddrDeviceBaseAddr);
        assert(B_off > this->_ddrDeviceBaseAddr);
        assert(C_off > this->_ddrDeviceBaseAddr);
        A_off -= this->_ddrDeviceBaseAddr;
        B_off -= this->_ddrDeviceBaseAddr;
        C_off -= this->_ddrDeviceBaseAddr;

        assert(A_off % this->PAGE_SIZE == 0);
        assert(B_off % this->PAGE_SIZE == 0);
        assert(C_off % this->PAGE_SIZE == 0);
        assert(X_off % this->PAGE_SIZE == 0);

        A_off /= this->PAGE_SIZE;
        B_off /= this->PAGE_SIZE;
        C_off /= this->PAGE_SIZE;
        X_off /= this->PAGE_SIZE;

        FcnArgs args(A_off, B_off, C_off, X_off, m,
                k, n, lda, ldb, ldc, ldx, postScale, postShift,  PReLUScale, PReLUAlpha);
        this->AddInstr ( &args);
#ifdef GEMX_PERF_DBG
        cout << "AddFCNOp: " << t.elapsed() << endl;
#endif
        return true;
    }

protected:
    //const int MIN_M = 256;
    //const int MIN_K = 256;
    //const int MIN_N = 32;

    bool isPowerOf2( int n )
    {
        return ( (n & (n-1)) == 0 );
    }

};
}
#endif
