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

extern "C" {

void MakeFCNHost(char *xclbin, char* device, unsigned int nPE);
void MakeGEMMHost(char *xclbin, char* device, unsigned int nPE);
void MakeSPMVHost(char *xclbin, char* device, unsigned int nPE);

void SendToFPGAShrt(short *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void SendToFPGAInt(int *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void SendToFPGAFloat(float *A,  unsigned long long num_elem, unsigned PE, bool sync_send);
void* SendSpToFpgaFloat(int *row, int *col, float *data, unsigned int nnz, unsigned int ddr_width, unsigned PE);
void* SendSpToFpgaInt(int *row, int *col, float *data, unsigned int nnz, unsigned int ddr_width, unsigned PE);
//void SendToFPGAShrt_dbg( char * name, short *A, int m, int n, bool sync_send);
//void SendToFPGAInt_dbg( char * name, int *A, int m, int n, bool sync_send);

void* GetFromFPGA( short *A, unsigned PE, bool sync_get);
void* GetFromFPGAInt( int *A, unsigned PE, bool sync_get);
void* GetFromFPGAFloat( float *A, unsigned PE, bool sync_get);
void Wait (unsigned PE);
void ClearInstrBuf (unsigned PE);
void PrintStats();
bool AddFCNOp( void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, short PReLUScale, short PReLUAlpha, unsigned PE);
bool AddGEMMOp( void * A, void * B, void *C, void * bias,  unsigned int m, unsigned int k, unsigned int n, int postScale, int postShift, unsigned PE);
bool AddSPMVOp(void *A, void * B, void *C, unsigned int m, unsigned int k, unsigned int nnz, unsigned PE);

int GetFreq ();
void Execute (bool sync_exec, unsigned PE);

void int16_gemm(short * A, short * B, short * X, short *C, unsigned int M, unsigned int K, unsigned int N );

}

