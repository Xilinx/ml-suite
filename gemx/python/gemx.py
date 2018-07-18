##################################################
#Copyright (c) 2018, Xilinx, Inc.
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification,
#are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice,
#this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#this list of conditions and the following disclaimer in the documentation
#and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors
#may be used to endorse or promote products derived from this software
#without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##################################################

from ctypes import *
import timeit
import numpy as np
import sys
import argparse
import math
class GEMXManager:
  def __init__(self, libFile):
    #self._handle = None
    
    self._lib = cdll.LoadLibrary(libFile)
    self._lib.MakeFCNHost.argtypes = [c_char_p, c_char_p,c_uint]
    self._lib.MakeGEMMHost.argtypes = [c_char_p, c_char_p, c_uint]
    self._lib.MakeSPMVHost.argtypes = [c_char_p, c_char_p, c_uint]                
    self._lib.SendToFPGAShrt.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), c_ulonglong, c_uint, c_bool]
    self._lib.SendToFPGAInt.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), c_ulonglong, c_uint, c_bool]
    self._lib.SendToFPGAFloat.argtypes = [np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_ulonglong, c_uint, c_bool]
    self._lib.AddFCNOp.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"),  
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), 
                                   c_uint, c_uint, c_uint, c_int, c_int, c_short, c_short, c_uint]
                                   
    self._lib.AddGEMMOp.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"),  
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), 
                                   c_uint, c_uint, c_uint, c_int, c_int, c_uint]
    self._lib.AddSPMVOp.argtypes = [c_void_p, 
                                   np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"), 
                                   np.ctypeslib.ndpointer(flags="C_CONTIGUOUS"),  
                                   c_uint, c_uint, c_uint, c_uint]                                                                 
    self._lib.SendSpToFpgaFloat.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),c_uint,c_uint,c_uint]
    self._lib.SendSpToFpgaInt.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"),
                                       np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),c_uint,c_uint,c_uint]
    self._lib.SendSpToFpgaFloat.restype = c_void_p
    self._lib.SendSpToFpgaInt.restype = c_void_p  
    self._lib.AddFCNOp.restype = c_bool
    self._lib.AddGEMMOp.restype = c_bool
    self._lib.AddSPMVOp.restype = c_bool
    self._lib.Execute.argtypes = [c_bool, c_uint]
    self._lib.GetFromFPGA.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), c_uint, c_bool]
    self._lib.GetFromFPGA.restype = c_void_p
    self._lib.GetFromFPGAInt.argtypes = [np.ctypeslib.ndpointer(c_int, flags="C_CONTIGUOUS"), c_uint, c_bool]
    self._lib.GetFromFPGAInt.restype = c_void_p
    self._lib.GetFromFPGAFloat.argtypes = [np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, c_bool]
    self._lib.GetFromFPGAFloat.restype = c_void_p
    self._lib.Wait.argtypes = [c_uint]
    self._lib.ClearInstrBuf.argtypes = [c_uint]
    
    self._lib.PrintStats.argtypes = []    
    self._lib.GetFreq.argtypes = []  
        
  def createFCNHandle (self, xclbin, deviceName, numHandles):
     b_xclbin = xclbin.encode('utf-8')
     b_device = deviceName.encode('utf-8')
     self._lib.MakeFCNHost(b_xclbin, b_device,int(numHandles))
     
  def createGEMMHandle (self, xclbin, deviceName, numHandles):
     b_xclbin = xclbin.encode('utf-8')
     b_device = deviceName.encode('utf-8')
     self._lib.MakeGEMMHost(b_xclbin, b_device, int(numHandles))
     
  def createSPMVHandle (self, xclbin, deviceName, numHandles):
     b_xclbin = xclbin.encode('utf-8')
     b_device = deviceName.encode('utf-8')
     self._lib.MakeSPMVHost(b_xclbin, b_device,int(numHandles))
     
  def addFCNOp(self, A, B, C, bias, postScale, postShift, PReLUScale, PReLUAlpha, PE):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Cannot perform FCN with matrices", A.shape, B.shape )
    if C.shape != bias.shape:
        raise ValueError("Bias matrix shape", bias.shape, "doesn't match output shape", C.shape)
    return self._lib.AddFCNOp(A,B, C, bias, c_uint(A.shape[0]), c_uint( A.shape[1] ), c_uint( B.shape[1]), c_int(postScale), c_int(postShift), c_short(PReLUScale), c_short(PReLUAlpha), c_uint(PE))
  
  def addGEMMOp(self, A, B, C, bias, postScale, postShift, PE):
    if A.shape[1] != B.shape[0]:
        raise ValueError("Cannot perform GEMM with matrices", A.shape, B.shape )
    if C.shape != bias.shape:
        raise ValueError("Bias matrix shape", bias.shape, "doesn't match output shape")      
    return self._lib.AddGEMMOp(A,B, C, bias, c_uint(A.shape[0]), c_uint( A.shape[1] ), c_uint( B.shape[1]), c_int(postScale), c_int(postShift), c_uint(PE))
  
  def addSPMVOp(self, A,B,C,nnz,PE=0):
    return self._lib.AddSPMVOp(A,B,C,c_uint(C.shape[0]),c_uint(B.shape[0]),c_uint(nnz),c_uint(PE))
    
  def execute(self, PE, sync_exec = True):
    self._lib.Execute(sync_exec, PE)
    
  def wait(self, PE):
    self._lib.Wait(PE)    
    
  def clearInstrBuf(self, PE):
    self._lib.ClearInstrBuf(PE)      
          
  def sendMat ( self, A, PE, sync_send = False):
    if A.flags['C_CONTIGUOUS'] == False:
        A = np.ascontiguousarray(A)
        print ("Warning: not C_CONTIGUOUS, performance will be affected")      
    if A.dtype == np.int32:
        self._lib.SendToFPGAInt( A, c_ulonglong(A.size), c_uint(PE), sync_send )
    elif A.dtype == np.int16:
        self._lib.SendToFPGAShrt( A, c_ulonglong(A.size), c_uint(PE), sync_send ) 
    elif A.dtype == np.float32:
        self._lib.SendToFPGAFloat( A, c_ulonglong(A.size), c_uint(PE), sync_send ) 
    else:
        raise TypeError("type", A.dtype, "not supported")
      
  def sendSpMat(self,row,col,data,ddrWidth,dtype,PE):
     if dtype == np.int32:
        return self._lib.SendSpToFpgaInt(row,col,data,row.shape[0],ddrWidth,c_uint(PE))   
     elif dtype == np.float32:     
        return self._lib.SendSpToFpgaFloat(row,col,data,row.shape[0],ddrWidth,c_uint(PE))
     else:
        raise TypeError("type", A, "not supported") 
          
  def getMat(self, A, PE=0, sync_get = True):
    if A.dtype == np.int16:
        self._lib.GetFromFPGA( A, PE, sync_get )
    elif A.dtype == np.int32:
        self._lib.GetFromFPGAInt( A, PE, sync_get )
    elif A.dtype == np.float32:
        self._lib.GetFromFPGAFloat( A, PE, sync_get )
    else:
        raise TypeError("type", A.dtype, "not supported") 
    
  def printStats(self):
    self._lib.PrintStats()
    
  def getFreq(self):
    return self._lib.GetFreq()

_gemxManager = None

def sendMat ( A,PE=0,sync_send=False):
    _gemxManager.sendMat(A,PE,sync_send)
    
def sendSpMat (row,col,data,ddrWidth,dtype,PE=0):
    return _gemxManager.sendSpMat(row,col,data,ddrWidth,dtype,PE)

def getMat (A, PE=0, sync_get = True):
    return _gemxManager.getMat(A, PE,sync_get)
    
def addFCNOp( A,B,C, bias, postScale, postShift, PReLUScale, PReLUAlpha,PE=0):
    _gemxManager.addFCNOp(A, B, C, bias, postScale, postShift, PReLUScale, PReLUAlpha, PE)
    
def addGEMMOp( A,B,C, bias, postScale, postShift,PE=0):
    _gemxManager.addGEMMOp(A, B, C, bias, postScale, postShift, PE)

def addSPMVOp( A,B,C,nnz,PE=0):
    _gemxManager.addSPMVOp(A,B,C,nnz,PE)
    
def execute(PE=0, sync_exec = True):
    _gemxManager.execute( PE, sync_exec)
    #_gemxManager.clearInstrBuf(PE)
    
def wait(PE=0):
    _gemxManager.wait(PE)    

def clearInstrBuf(PE=0):
    _gemxManager.clearInstrBuf(PE)    
      
def createManager ( libFile ):
  global _gemxManager
  if not _gemxManager:
    _gemxManager = GEMXManager(libFile)    
  return True  
    
def createFCNHandle(args, xclbin_opts):
  createManager (args.gemxlib)
  return _gemxManager.createFCNHandle(args.xclbin, xclbin_opts["GEMX_part"], xclbin_opts["GEMX_numKernels"])

def createGEMMHandle(args, xclbin_opts):
  createManager (args.gemxlib)
  return _gemxManager.createGEMMHandle(args.xclbin, xclbin_opts["GEMX_part"], xclbin_opts["GEMX_numKernels"])
  
def createSPMVHandle(args, xclbin_opts):
  createManager (args.gemxlib)
  return _gemxManager.createSPMVHandle(args.xclbin, xclbin_opts["GEMX_part"], xclbin_opts["GEMX_numKernels"])

def printStats():
  return _gemxManager.printStats()
  
def getFreq():
  return _gemxManager.getFreq()

def create_fpga_buf ( shape, np_type , PE=0):
    a = np.zeros ( shape, dtype=np_type, order='C')
    _gemxManager.sendMat(a, PE)
    return a

def parse_cfg(filename):
    myvars = {}
    with open(filename) as myfile:
        for line in myfile:
            for word in line.split():
               name, var = word.split("=")
               myvars[name.strip()] = var.rstrip()  
    return myvars

def default_args():
  parser = argparse.ArgumentParser(description='GEMX')
  parser.add_argument('--xclbin', required = True, help='file path to FPGA bitstream')
  parser.add_argument('--gemxlib', required = True, help='file path to GEMX host code shared library')
  parser.add_argument('--cfg', required=True, help='file describing .xclbin properties')
  return parser
    
def processCommandLine():
  parser = default_args()
  args = parser.parse_args()
  xclbin_opts = parse_cfg ( args.cfg ) 
  return args, xclbin_opts
