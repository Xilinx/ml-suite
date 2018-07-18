/* autopilot_ssdm_op.h*/
/*
 * __VIVADO_HLS_COPYRIGHT-INFO__ 
 *
 * $Id$
 */

#ifndef _AUTOPILOT_SSDM_OP_H_
#define _AUTOPILOT_SSDM_OP_H_

#ifdef AESL_SYN
#define ap_wait() _ssdm_op_Wait(1)
#define ap_wait_n(X) {if(X<=1){ _ssdm_op_Wait(1); return; } \
    for(unsigned __i__=0; __i__<X; ++__i__) { \
        _ssdm_Unroll(0,0,0, ""); \
        _ssdm_op_Wait(1); \
    } \
}
#define ap_wait_until(X) { _ssdm_op_Wait(1); int __t = X; _ssdm_op_Poll(__t); }

#define SSDM_SPEC_FIFO(M, TY, DT, C, D)    \
    _ssdm_op_SpecChannel(#M, _ssdm_sc_fifo, #DT, #C, D, D, &C, &C);   \
    _ssdm_op_SpecInterface(&C, "ap_fifo", 0, 0, "", 0, 0, "", "", "", 0, 0, 0, 0, "", "")

#ifndef SSDM_KEEP_NAME
#define SSDM_KEEP_NAME(__name__) _ssdm_op_SpecExt("member_name", #__name__, &__name__);
#define SSDM_KEEP_name(__name__, __reference__) _ssdm_op_SpecExt("member_name", #__name__, __reference__);
#define SSDM_RENAME(__name__, __newname__) _ssdm_op_SpecExt("member_name", #__newname__, &__name__);
#endif

#else

#define ap_wait() 
#define ap_wait_n(X) 
#define ap_wait_until(X) do { ap_wait(); } while (!(X));

#ifndef SSDM_KEEP_NAME
#define SSDM_KEEP_NAME(__name__) 
#endif

#endif

#ifdef AUTOPILOT_SCPP_SYSC_SIM

/**************************************************************
** SystemC simulation constructs.
***************************************************************/
#include "systemc"
using namespace sc_dt;

#define _ssdm_op_GetRange(V, u, l) (V.range(u, l))
#define _ssdm_op_SetRange(V, u, l, x)  (V.range(u, l) = x)
#define _ssdm_op_GetBit(V, pos) (V[pos])
#define _ssdm_op_SetBit(V, pos, x)  (V[pos] = x)

#define _ssdm_op_BuffRead(buff) (*(buff))
#define _ssdm_op_BuffWrite(buff, data) ((*(buff)) = data)
#define _ssdm_op_FifoRead(f) (*(f))
#define _ssdm_op_FifoWrite(f, data) ((*(f)) = data)
#define _ssdm_op_WireRead(f) (*(f))
#define _ssdm_op_WireWrite(f, data) ((*(f)) = data)

#define _ssdm_FifoRead(f) _ssdm_op_FifoRead(f)
#define _ssdm_FifoWrite(fifo, data) _ssdm_op_FifoWrite(fifo, data)
#define _ssdm_op_read(f, x) (*(f));  _ssdm_op_Wait(1)
#define _ssdm_op_write(f, x) ((*(f)) = data); _ssdm_op_Wait(1)


#define _ssdm_WaitEvent(e)  _ssdm_op_Wait(e)

#define _ssdm_op_SpecPipeline(x) 
#define _ssdm_op_SpecDataflowPipeline(x) 
#define _ssdm_op_Wait(x) 
#define _ssdm_op_Poll(x) 

#define SSDM_SPEC_MODULE(X) 
#define SSDM_SPEC_PROCESS(M, TY, X) 
#define SSDM_SPEC_PORT(M, TY, DT, P) 
#define SSDM_SPEC_CONNECTION(M, X, P) 
#define SSDM_SPEC_CHANNEL(M, TY, DT, C) 
#define SSDM_SPEC_SENSITIVE(M, S, TY)


#else


#ifdef AUTOPILOT_BC_SIM

/**************************************************************
** BC simulation constructs.
***************************************************************/

#define _ssdm_WaitEvent(e)  _ssdm_op_Wait(e)

#else

#if defined(__cplusplus) || defined(__openclc)

/*#define AP_SPEC_ATTR __attribute__ ((pure))*/
//adu: patched
#if (__clang_major__ == 3) && (__clang_minor__ == 9)
#define SSDM_SPEC_ATTR __attribute__ ((nothrow)) __attribute__((overloadable))
#define SSDM_OP_ATTR __attribute__ ((nothrow)) __attribute__((overloadable))
#else
#define SSDM_SPEC_ATTR __attribute__ ((nothrow))
#define SSDM_OP_ATTR __attribute__ ((nothrow))
#endif

#ifdef __cplusplus
extern "C" {
#endif
    /****** SSDM Intrinsics: OPERATIONS ***/
    // Interface operations

    //typedef unsigned int __attribute__ ((bitwidth(1))) _uint1_;
    typedef bool _uint1_;

    void _ssdm_op_IfRead(...) SSDM_OP_ATTR;
    void _ssdm_op_IfWrite(...) SSDM_OP_ATTR;
    _uint1_ _ssdm_op_IfNbRead(...) SSDM_OP_ATTR;
    _uint1_ _ssdm_op_IfNbWrite(...) SSDM_OP_ATTR;
    _uint1_ _ssdm_op_IfCanRead(...) SSDM_OP_ATTR;
    _uint1_ _ssdm_op_IfCanWrite(...) SSDM_OP_ATTR;

    // Stream Intrinsics
    void _ssdm_StreamRead(...) SSDM_OP_ATTR;
    void _ssdm_StreamWrite(...) SSDM_OP_ATTR;
    _uint1_  _ssdm_StreamNbRead(...) SSDM_OP_ATTR;
    _uint1_  _ssdm_StreamNbWrite(...) SSDM_OP_ATTR;
    _uint1_  _ssdm_StreamCanRead(...) SSDM_OP_ATTR;
    _uint1_  _ssdm_StreamCanWrite(...) SSDM_OP_ATTR;
    unsigned _ssdm_StreamSize(...) SSDM_OP_ATTR;

    // Misc
    void _ssdm_op_MemShiftRead(...) SSDM_OP_ATTR;

    void _ssdm_op_Wait(...) SSDM_OP_ATTR;
    void _ssdm_op_Poll(...) SSDM_OP_ATTR;

    void _ssdm_op_Return(...) SSDM_OP_ATTR;

    /* SSDM Intrinsics: SPECIFICATIONS */
    void _ssdm_op_SpecSynModule(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecTopModule(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecProcessDecl(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecProcessDef(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecPort(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecConnection(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecChannel(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecSensitive(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecModuleInst(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecPortMap(...) SSDM_SPEC_ATTR;

    void _ssdm_op_SpecReset(...) SSDM_SPEC_ATTR;

    void _ssdm_op_SpecPlatform(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecClockDomain(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecPowerDomain(...) SSDM_SPEC_ATTR;
                                   
    int _ssdm_op_SpecRegionBegin(...) SSDM_SPEC_ATTR;    
    int _ssdm_op_SpecRegionEnd(...) SSDM_SPEC_ATTR;

    void _ssdm_op_SpecLoopName(...) SSDM_SPEC_ATTR;    
    
    void _ssdm_op_SpecLoopTripCount(...) SSDM_SPEC_ATTR;

    int _ssdm_op_SpecStateBegin(...) SSDM_SPEC_ATTR;
    int _ssdm_op_SpecStateEnd(...) SSDM_SPEC_ATTR;

    void _ssdm_op_SpecInterface(...) SSDM_SPEC_ATTR;

    void _ssdm_op_SpecPipeline(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecDataflowPipeline(...) SSDM_SPEC_ATTR;


    void _ssdm_op_SpecLatency(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecParallel(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecProtocol(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecOccurrence(...) SSDM_SPEC_ATTR;

    void _ssdm_op_SpecResource(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecResourceLimit(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecCHCore(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecFUCore(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecIFCore(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecIPCore(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecKeepValue(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecMemCore(...) SSDM_SPEC_ATTR;

    void _ssdm_op_SpecExt(...) SSDM_SPEC_ATTR;
    /*void* _ssdm_op_SpecProcess(...) SSDM_SPEC_ATTR;
    void* _ssdm_op_SpecEdge(...) SSDM_SPEC_ATTR; */
   
    /* Presynthesis directive functions */
    void _ssdm_SpecArrayDimSize(...) SSDM_SPEC_ATTR;

    void _ssdm_RegionBegin(...) SSDM_SPEC_ATTR;
    void _ssdm_RegionEnd(...) SSDM_SPEC_ATTR;

    void _ssdm_Unroll(...) SSDM_SPEC_ATTR;
    void _ssdm_UnrollRegion(...) SSDM_SPEC_ATTR;

    void _ssdm_InlineAll(...) SSDM_SPEC_ATTR;
    void _ssdm_InlineLoop(...) SSDM_SPEC_ATTR;
    void _ssdm_Inline(...) SSDM_SPEC_ATTR;
    void _ssdm_InlineSelf(...) SSDM_SPEC_ATTR;
    void _ssdm_InlineRegion(...) SSDM_SPEC_ATTR;

    void _ssdm_SpecArrayMap(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecArrayPartition(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecArrayReshape(...) SSDM_SPEC_ATTR;

    void _ssdm_SpecStream(...) SSDM_SPEC_ATTR;
    
    void _ssdm_SpecExpr(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecExprBalance(...) SSDM_SPEC_ATTR;

    void _ssdm_SpecDependence(...) SSDM_SPEC_ATTR;
                                    
    void _ssdm_SpecLoopMerge(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecLoopFlatten(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecLoopRewind(...) SSDM_SPEC_ATTR;                                

    void _ssdm_SpecFuncInstantiation(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecFuncBuffer(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecFuncExtract(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecConstant(...) SSDM_SPEC_ATTR;
    
    void _ssdm_DataPack(...) SSDM_SPEC_ATTR;
    void _ssdm_SpecDataPack(...) SSDM_SPEC_ATTR;

    void _ssdm_op_SpecBitsMap(...) SSDM_SPEC_ATTR;
    void _ssdm_op_SpecLicense(...) SSDM_SPEC_ATTR;
#ifndef AESL_TB
    void __xilinx_ip_top(...) SSDM_SPEC_ATTR;
#endif
#ifdef __cplusplus
}
#endif

#else

/*#define AP_SPEC_ATTR __attribute__ ((pure))*/
#define SSDM_SPEC_ATTR __attribute__ ((nothrow))
#define SSDM_OP_ATTR __attribute__ ((nothrow))

    /****** SSDM Intrinsics: OPERATIONS ***/
    // Interface operations
    //typedef unsigned int __attribute__ ((bitwidth(1))) _uint1_;
    void _ssdm_op_IfRead() SSDM_OP_ATTR;
    void _ssdm_op_IfWrite() SSDM_OP_ATTR;
    //_uint1_ _ssdm_op_IfNbRead() SSDM_OP_ATTR;
    //_uint1_ _ssdm_op_IfNbWrite() SSDM_OP_ATTR;
    //_uint1_ _ssdm_op_IfCanRead() SSDM_OP_ATTR;
    //_uint1_ _ssdm_op_IfCanWrite() SSDM_OP_ATTR;

    // Stream Intrinsics
    void _ssdm_StreamRead() SSDM_OP_ATTR;
    void _ssdm_StreamWrite() SSDM_OP_ATTR;
    //_uint1_  _ssdm_StreamNbRead() SSDM_OP_ATTR;
    //_uint1_  _ssdm_StreamNbWrite() SSDM_OP_ATTR;
    //_uint1_  _ssdm_StreamCanRead() SSDM_OP_ATTR;
    //_uint1_  _ssdm_StreamCanWrite() SSDM_OP_ATTR;

    // Misc
    void _ssdm_op_MemShiftRead() SSDM_OP_ATTR;

    void _ssdm_op_Wait() SSDM_OP_ATTR;
    void _ssdm_op_Poll() SSDM_OP_ATTR;

    void _ssdm_op_Return() SSDM_OP_ATTR;

    /* SSDM Intrinsics: SPECIFICATIONS */
    void _ssdm_op_SpecSynModule() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecTopModule() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecProcessDecl() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecProcessDef() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecPort() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecConnection() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecChannel() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecSensitive() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecModuleInst() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecPortMap() SSDM_SPEC_ATTR;

    void _ssdm_op_SpecReset() SSDM_SPEC_ATTR;

    void _ssdm_op_SpecPlatform() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecClockDomain() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecPowerDomain() SSDM_SPEC_ATTR;
                                   
    int _ssdm_op_SpecRegionBegin() SSDM_SPEC_ATTR;    
    int _ssdm_op_SpecRegionEnd() SSDM_SPEC_ATTR;

    void _ssdm_op_SpecLoopName() SSDM_SPEC_ATTR;    
    
    void _ssdm_op_SpecLoopTripCount() SSDM_SPEC_ATTR;

    int _ssdm_op_SpecStateBegin() SSDM_SPEC_ATTR;
    int _ssdm_op_SpecStateEnd() SSDM_SPEC_ATTR;

    void _ssdm_op_SpecInterface() SSDM_SPEC_ATTR;

    void _ssdm_op_SpecPipeline() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecDataflowPipeline() SSDM_SPEC_ATTR;


    void _ssdm_op_SpecLatency() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecParallel() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecProtocol() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecOccurrence() SSDM_SPEC_ATTR;

    void _ssdm_op_SpecResource() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecResourceLimit() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecCHCore() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecFUCore() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecIFCore() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecIPCore() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecKeepValue() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecMemCore() SSDM_SPEC_ATTR;

    void _ssdm_op_SpecExt() SSDM_SPEC_ATTR;
    /*void* _ssdm_op_SpecProcess() SSDM_SPEC_ATTR;
    void* _ssdm_op_SpecEdge() SSDM_SPEC_ATTR; */
   
    /* Presynthesis directive functions */
    void _ssdm_SpecArrayDimSize() SSDM_SPEC_ATTR;

    void _ssdm_RegionBegin() SSDM_SPEC_ATTR;
    void _ssdm_RegionEnd() SSDM_SPEC_ATTR;

    void _ssdm_Unroll() SSDM_SPEC_ATTR;
    void _ssdm_UnrollRegion() SSDM_SPEC_ATTR;

    void _ssdm_InlineAll() SSDM_SPEC_ATTR;
    void _ssdm_InlineLoop() SSDM_SPEC_ATTR;
    void _ssdm_Inline() SSDM_SPEC_ATTR;
    void _ssdm_InlineSelf() SSDM_SPEC_ATTR;
    void _ssdm_InlineRegion() SSDM_SPEC_ATTR;

    void _ssdm_SpecArrayMap() SSDM_SPEC_ATTR;
    void _ssdm_SpecArrayPartition() SSDM_SPEC_ATTR;
    void _ssdm_SpecArrayReshape() SSDM_SPEC_ATTR;

    void _ssdm_SpecStream() SSDM_SPEC_ATTR;
    
    void _ssdm_SpecExpr() SSDM_SPEC_ATTR;
    void _ssdm_SpecExprBalance() SSDM_SPEC_ATTR;

    void _ssdm_SpecDependence() SSDM_SPEC_ATTR;
                                    
    void _ssdm_SpecLoopMerge() SSDM_SPEC_ATTR;
    void _ssdm_SpecLoopFlatten() SSDM_SPEC_ATTR;
    void _ssdm_SpecLoopRewind() SSDM_SPEC_ATTR;                                

    void _ssdm_SpecFuncInstantiation() SSDM_SPEC_ATTR;
    void _ssdm_SpecFuncBuffer() SSDM_SPEC_ATTR;
    void _ssdm_SpecFuncExtract() SSDM_SPEC_ATTR;
    void _ssdm_SpecConstant() SSDM_SPEC_ATTR;
    
    void _ssdm_DataPack() SSDM_SPEC_ATTR;
    void _ssdm_SpecDataPack() SSDM_SPEC_ATTR;

    void _ssdm_op_SpecBitsMap() SSDM_SPEC_ATTR;
    void _ssdm_op_SpecLicense() SSDM_SPEC_ATTR;
#endif

/*#define _ssdm_op_WaitUntil(X) while (!(X)) _ssdm_op_Wait(1);
#define _ssdm_op_Delayed(X) X */

#define _ssdm_op_BuffRead(buff) (*(buff))
#define _ssdm_op_BuffWrite(buff, data) (*(buff) = data)
#define _ssdm_op_WireRead(buff) (*(buff))
#define _ssdm_op_WireWrite(buff, data) (*(buff) = data)

#endif

#endif

#endif

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
