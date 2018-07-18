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

#ifndef _SPMV_HOST_H_
#define _SPMV_HOST_H_

#include "xhost.h"
#include "gemm_host.h"
#include "gemx_util.h"

using namespace std;
namespace gemx {

template < typename Tdata,  typename Tidx>
class SpMatUram
{
  private:
    unsigned int m_Nnz;
    Tdata *m_DataAddr;
    Tidx  *m_IdxAddr;
    unsigned int t_NumData;
    unsigned int t_NumIdx;
    unsigned int m_ddrWidth;
  public:
    SpMatUram(){}
    SpMatUram(unsigned int p_Nnz, Tdata *p_DataAddr, unsigned int ddr_width)
      : m_Nnz(p_Nnz), m_DataAddr(p_DataAddr), m_IdxAddr((Tidx*)(p_DataAddr+ddr_width)), m_ddrWidth(ddr_width) {
      t_NumData = (sizeof(Tidx)/sizeof(Tdata)) * 2 * ddr_width + ddr_width;
      t_NumIdx = (sizeof(Tidx)*2/sizeof(Tdata)+1)*sizeof(Tdata)*ddr_width / sizeof(Tidx);
    }
    inline Tdata &getVal(unsigned int p_id) {return m_DataAddr[(p_id/m_ddrWidth)*t_NumData+(p_id%m_ddrWidth)];}
    inline Tidx &getCol(unsigned int p_id) {return m_IdxAddr[(p_id/m_ddrWidth)*t_NumIdx+(p_id%m_ddrWidth)*2];}
    inline Tidx &getRow(unsigned int p_id) {return m_IdxAddr[(p_id/m_ddrWidth)*t_NumIdx+(p_id%m_ddrWidth)*2+1];}
    
    void
    fillFromVector(int* row, int* col, float* data) { 
      for (unsigned int i = 0; i < m_Nnz; ++i) {
        getVal(i) = data[i];
        getCol(i) = col[i];
        getRow(i) = row[i]; 
      }
    }
    
};
  
class SpmvArgs: public kArgs {
public:
    virtual ~SpmvArgs() {
    }
    SpmvArgs() = delete;
    SpmvArgs ( unsigned int p_Aoffset, unsigned int p_Boffset, unsigned int p_Coffset, unsigned int M, unsigned int K, unsigned int Nnz) :
        m_spmv_args( { int(OpSpmv), p_Aoffset, p_Boffset, p_Coffset, M, K, Nnz, 0, 0, 0, 0, 0, 0, 0, 0, 0} ){
    }

    size_t sizeInBytes() {
        return sizeof(m_spmv_args);
    }
    char *asByteArray() {
        return reinterpret_cast<char*>(&m_spmv_args);
    }
protected:
    struct {
        int m_optype;
        unsigned int m_Aoffset, m_Boffset, m_Coffset, m_M, m_K, m_Nnz;
        unsigned int dummy[9];
    } m_spmv_args;
};

template<typename HType>
class SPMVHost : public GEMMHost<HType> {
public:
    SPMVHost() = delete;
    virtual ~SPMVHost(){
    }

    SPMVHost(const SPMVHost<HType> &) = delete;

    SPMVHost(const string & xclbin, const string & kernelName, const unsigned ddrBank, const string & device) : GEMMHost<HType> ( xclbin, kernelName, ddrBank, device)
    {
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType &C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift) {
        cerr << "GEMM operation not supported" << endl;
        return false;
    }

    virtual bool AddGEMMOp(const HType & A, const HType & B, const HType & C, const HType & bias, unsigned int m, unsigned int k, unsigned int n, unsigned int lda, unsigned int ldb, unsigned int ldc, unsigned int ldx, int postScale, int postShift) {
        cerr << "GEMM operation not supported" << endl;
        return false;
    } 
    
    virtual void* SendSpToFpgaFloat(int * row, int * col, float * data, unsigned int nnz, unsigned int ddr_width){
       float *A = new float[nnz+nnz*2*sizeof(int)/sizeof(float)];
       SpMatUram<float,int> MatA(nnz,A,ddr_width);
       MatA.fillFromVector(row,col,data);
       this->SendToFPGA((float*)A, A,(unsigned long long)(nnz+nnz*2*sizeof(int)/sizeof(float))*sizeof(float)); 
       return A;
    }
    
    virtual void* SendSpToFpgaInt(int * row, int * col, float * data, unsigned int nnz,unsigned int ddr_width){
       int *A =new int[nnz+nnz*2*sizeof(int)/sizeof(int)];
       SpMatUram<int,int> MatA(nnz,A,ddr_width);
       MatA.fillFromVector(row,col,data);
       this->SendToFPGA((float*)A, A,(unsigned long long)(nnz+nnz*2*sizeof(int)/sizeof(int))*sizeof(int));     
       return A;
    }
    
    virtual bool AddSPMVOp(const HType & A, const HType & B, const HType & C, unsigned int m, unsigned int k, unsigned int nnz){     
      if (this->_hostMat.find(A) == this->_hostMat.end()
                || this->_hostMat.find(B) == this->_hostMat.end()
                || this->_hostMat.find(C) == this->_hostMat.end()) {
            cerr << "Matrix not found!" << endl;
            return false;
        }
       
       unsigned long long A_off = 0, B_off = 0, C_off = 0;
       xclGetMemObjDeviceAddress(this->_devHandle[A].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &A_off);
       xclGetMemObjDeviceAddress(this->_devHandle[B].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &B_off);
       xclGetMemObjDeviceAddress(this->_devHandle[C].get(),
                boost::compute::system::default_device().get(),
                sizeof(unsigned long long), &C_off);
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

       A_off /= this->PAGE_SIZE;
       B_off /= this->PAGE_SIZE;
       C_off /= this->PAGE_SIZE;

       SpmvArgs args(A_off, B_off, C_off, m, k, nnz);
       this->AddInstr (&args);  
       return true;
    }
       
};

}


#endif
