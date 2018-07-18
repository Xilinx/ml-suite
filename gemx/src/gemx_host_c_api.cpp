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

#include <iostream>
#include <fstream>
#include <unordered_map>
#include "gemm_host.h"
#include "fcn_host.h"
#include "spmv_host.h"
#include "xhost.h"
#include "gemx_util.h"
#include "gemx_host_c_api.h"

//#define GEMX_PERF_DBG
using namespace gemx;
using namespace std;


class GEMXHostProfiler {
public:
    unordered_map < string, double> func_time;
    unordered_map < string, unsigned long long> func_calls;
    static GEMXHostProfiler& Instance() {
        static GEMXHostProfiler theInstance;
        return theInstance;
    }
protected:
    GEMXHostProfiler() {

    }
};
template<typename HType>
class GEMXHostHandle {
public:
    vector<shared_ptr<GEMMHost<HType>>> gh_ptr;
    static GEMXHostHandle& Instance() {
        static GEMXHostHandle theInstance;
        return theInstance;
    }
protected:
    GEMXHostHandle() {

    }
};


template<typename T>
static void print(char *name,T * A, int m, int n)
{
    ofstream myfile;
    string fName = name;
    fName += ".c";
    myfile.open (fName.c_str());

    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            myfile << A[i*m + j] << " ";
        }
        myfile << "\n";
    }
    myfile.close();
}

void MakeFCNHost(char *xclbin, char * device, unsigned int nPE)
{
    vector<unsigned> ddr = GEMMHost<short*>::getDDRBankFlags(device);
    for (unsigned i = 0; i < nPE; i++)
    {
        string kName = GEMMHost<short*>::getKernelName(i);
        GEMXHostHandle<void*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<void*> > (new gemx::FCNHost<void*>(xclbin, kName, ddr[i], device) ));
    }
}

void MakeGEMMHost(char *xclbin, char * device, unsigned int nPE)
{
	vector<unsigned> ddr = GEMMHost<short*>::getDDRBankFlags(device);
	for (unsigned i = 0; i < nPE; i++)
	{
		string kName = GEMMHost<short*>::getKernelName(i);
    	GEMXHostHandle<void*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<void*> > ( new gemx::GEMMHost<void*>(xclbin, kName, ddr[i], device) ));
	}
}

void MakeSPMVHost(char *xclbin, char * device, unsigned int nPE) {
  
  	vector<unsigned> ddr = GEMMHost<void*>::getDDRBankFlags(device);
	for (unsigned i = 0; i < nPE; i++)
	{
		string kName = GEMMHost<void*>::getKernelName(i);
    	GEMXHostHandle<void*>::Instance().gh_ptr.push_back(shared_ptr< gemx::GEMMHost<void*> > ( new gemx::SPMVHost<void*>(xclbin, kName, ddr[i], device) ));
	}
}

void SendToFPGAShrt(short *A, unsigned long long num_elem, unsigned PE, bool sync_send)
{
    gemx::XTimer t;

    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->SendToFPGA(A, A, sizeof(short) *num_elem, sync_send);
    //SendToFPGA( A, sizeof(short) * num_elem, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAShrt"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAShrt"]++;
#endif
}

void SendToFPGAInt(int *A, unsigned long long num_elem, unsigned PE,bool sync_send)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->SendToFPGA(A, A, sizeof(int) *num_elem, sync_send);
    //SendToFPGA(A, sizeof(int) *num_elem, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAInt"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAInt"]++;
#endif

}

void SendToFPGAFloat(float *A, unsigned long long num_elem, unsigned PE,bool sync_send)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->SendToFPGA(A, A, sizeof(float) *num_elem, sync_send);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendToFPGAFloat"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendToFPGAFloat"]++;
#endif

}

void* SendSpToFpgaFloat(int *row, int *col, float *data, unsigned int nnz, unsigned int ddr_width, unsigned PE){
    gemx::XTimer t;
    gemx::SPMVHost<void*>* spmv_ptr = static_cast< gemx::SPMVHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    void* ret = spmv_ptr->SendSpToFpgaFloat(row,col,data,nnz,ddr_width);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendSpToFpgaFloat"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendSpToFpgaFloat"]++;
#endif
    return ret;
}

void* SendSpToFpgaInt(int *row, int *col, float *data, unsigned int nnz, unsigned int ddr_width, unsigned PE){
    gemx::XTimer t;
    gemx::SPMVHost<void*>* spmv_ptr = static_cast< gemx::SPMVHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    void* ret = spmv_ptr->SendSpToFpgaInt(row,col,data,nnz,ddr_width);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["SendSpToFpgaInt"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["SendSpToFpgaInt"]++;
#endif
    return ret;
}

/*
void SendToFPGAShrt_dbg(char * name, short *A, int m, int n, bool sync_send){
    print<short>(name, A, m,n);
    GEMXHostHandle<short*>::Instance().gh_ptr->SendToFPGA(name, A, sizeof(short) * m * n, sync_send);
}

void SendToFPGAInt_dbg(char * name, int *A, int m, int n, bool sync_send){
    print<int>(name, A, m,n);
    GEMXHostHandle<short*>::Instance().gh_ptr->SendToFPGA(name, A, sizeof(int) * m * n, sync_send);
}
*/

void* GetFromFPGA(short *A, unsigned PE, bool sync_get)
{
    gemx::XTimer t;
    void * ptr = GEMXHostHandle<void*>::Instance().gh_ptr[PE]->GetMat(A, true, sync_get);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["GetFromFPGA"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["GetFromFPGA"]++;
#endif
    return ptr;
}

void* GetFromFPGAInt(int *A, unsigned PE, bool sync_get)
{
    gemx::XTimer t;
    void * ptr = GEMXHostHandle<void*>::Instance().gh_ptr[PE]->GetMat(A, true, sync_get);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["GetFromFPGA"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["GetFromFPGA"]++;
#endif
    return ptr;
}

void* GetFromFPGAFloat(float *A, unsigned PE, bool sync_get)
{
    gemx::XTimer t;
    void * ptr = GEMXHostHandle<void*>::Instance().gh_ptr[PE]->GetMat(A, true, sync_get);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["GetFromFPGA"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["GetFromFPGA"]++;
#endif
    return ptr;
}

bool AddFCNOp(void * A, void * B, void *C, void * bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha,unsigned PE)
{
    gemx::XTimer t;
    //cout << C << " = " << A << " * " << B << " + " << bias << endl;
    gemx::FCNHost<void*>* fcn_ptr = static_cast< gemx::FCNHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    bool ret = fcn_ptr->AddFCNOp(A, B, C, bias, m,k,n, postScale, postShift, PReLUScale, PReLUAlpha);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["AddFCNOp"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["AddFCNOp"]++;
#endif
    return ret;
}

bool AddGEMMOp(void * A, void * B, void *C, void * bias, unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, unsigned PE)
{
    //cout << C << " = " << A << " * " << B << " + " << bias << endl;
    return GEMXHostHandle<void*>::Instance().gh_ptr[PE]->AddGEMMOp(A, B, C, bias, m,k,n, postScale, postShift);
}

bool AddSPMVOp(void *A, void * B, void *C, unsigned int m, unsigned int k, unsigned int nnz, unsigned PE)
{
    gemx::SPMVHost<void*>* spmv_ptr = static_cast< gemx::SPMVHost<void*> *> (GEMXHostHandle<void*>::Instance().gh_ptr[PE].get());
    bool ret = spmv_ptr->AddSPMVOp(A, B, C, m, k, nnz);
    return ret;
}

void Execute (bool sync_exec, unsigned PE)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->Execute(sync_exec);
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["Execute"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["Execute"]++;
#endif

}

void Wait (unsigned PE)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->Wait();
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["Wait"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["Wait"]++;
#endif
}

void ClearInstrBuf(unsigned PE)
{
    gemx::XTimer t;
    GEMXHostHandle<void*>::Instance().gh_ptr[PE]->ClearInstrBuf();
#ifdef GEMX_PERF_DBG
    GEMXHostProfiler::Instance().func_time["ClearInstrBuf"] += t.elapsed();
    GEMXHostProfiler::Instance().func_calls["ClearInstrBuf"]++;
#endif
}

void PrintStats()
{
    for ( auto p : GEMXHostProfiler::Instance().func_time)
    {
        cout << p.first << ": " << (p.second * 1000.0) / GEMXHostProfiler::Instance().func_calls[p.first]  << " ms " << GEMXHostProfiler::Instance().func_calls[p.first] << " calls" << endl;
    }
}

int GetFreq(){
  return GEMXHostHandle<void*>::Instance().gh_ptr[0]->getBoardFreqMHz(0);
}

//void DestroyGEMMHost(gemx::GEMMHost<short*> * ptr) {
//    delete ptr;
//}
void int16_gemm(short * A, short * B, short *X,
        short * C, unsigned int M, unsigned int K, unsigned int N ) {
    using namespace std;
    using namespace gemx;

    cout << "A_ptr: " << A << " B_ptr: " << B << " C_ptr: " << C << " X_ptr: " << X << endl;
    shared_ptr<GEMMHost<void*>> host_ptr = GEMXHostHandle<void*>::Instance().gh_ptr[0];

    host_ptr->SendToFPGA((short*)A, A, sizeof(short)*M*K);
    host_ptr->SendToFPGA((short*)B, B, sizeof(short)*K*N);
    host_ptr->SendToFPGA((short*)C, C, sizeof(short)*M*N);
    host_ptr->SendToFPGA((short*)X, X, sizeof(short)*M*N);

    host_ptr->AddGEMMOp(A, B, C, (short*)X, M,K,N, 1,0);

    host_ptr->Execute();

    host_ptr->GetMat(C, true, true);
}
