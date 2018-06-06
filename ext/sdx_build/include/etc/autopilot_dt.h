/*-*-c++-*-*/
/* autopilot_dt.h: defines all bit-accurate data types.*/
/*
 * __VIVADO_HLS_COPYRIGHT-INFO__ 
 *
 * $Id$
 */

#ifndef _AUTOPILOT_DT_H_
#define _AUTOPILOT_DT_H_
#ifndef AP_INT_MAX_W
#define AP_INT_MAX_W 1024
#endif
#if AP_INT_MAX_W > (1 << 15) 
#error "Bitwidth exceeds 32768 (1 << 15), the maximum allowed value"
#endif
#define AP_INT_MAX_UNIT ((AP_INT_MAX_W + 1023) / 1024)
#define _TY_CONCAT(P, T) P##T

#if defined(__cplusplus) & defined(C_TEST)
#  warning "Using SystemC data types"

#  include "systemc.h"
using namespace sc_dt;

#if defined(APINT_TYPE_PREFIX)
#  define HANDLE_BAT(T, N) typedef sc_dt::sc_int<N> _TY_CONCAT(T, int##N);
#else
#  define HANDLE_BAT(T, N) typedef sc_dt::sc_int<N> _TY_CONCAT(, int##N);
#endif
#  include "etc/autopilot_dt.def"

#if defined(APINT_TYPE_PREFIX)
#  define HANDLE_BAT(T, N) typedef sc_dt::sc_uint<N> _TY_CONCAT(T, uint##N);
#else
#  define HANDLE_BAT(T, N) typedef sc_dt::sc_uint<N> _TY_CONCAT(, uint##N);
#endif
#  include "etc/autopilot_dt.def"

/* No definition for SystemC sc_dt::int64 and uint64.*/

#else

#if defined(APINT_TYPE_PREFIX)
#  define HANDLE_BAT(T, N) \
    typedef int __attribute__ ((bitwidth(N))) _TY_CONCAT(T, int##N);
#else
#  define HANDLE_BAT(T, N) \
    typedef int __attribute__ ((bitwidth(N))) _TY_CONCAT(, int##N);
#endif
#  include "etc/autopilot_dt.def"
#  include "etc/autopilot_dt_ext.def"
#undef HANDLE_BAT

#if defined(APINT_TYPE_PREFIX)
#  define HANDLE_BAT(T, N) \
    typedef unsigned int __attribute__ ((bitwidth(N))) _TY_CONCAT(T, uint##N);
#else
#  define HANDLE_BAT(T, N) \
    typedef unsigned int __attribute__ ((bitwidth(N))) _TY_CONCAT(, uint##N);
#endif
#  include "etc/autopilot_dt.def"
#  include "etc/autopilot_dt_ext.def"
#undef HANDLE_BAT

#undef AP_INT_MAX_UNIT

#  if defined(__cplusplus)
/*#    warning "Incomplete C++ data types"
#    include "autopilot_dt.hpp" */
#    ifndef __SC_COMPATIBLE__
         typedef int __attribute__ ((bitwidth(64))) ap_int64;
         typedef unsigned int __attribute__ ((bitwidth(64))) ap_uint64;
#        define  int64  ap_int64
#        define uint64 ap_uint64
#    endif

#  else

#if defined(APINT_TYPE_PREFIX)
typedef int __attribute__ ((bitwidth(64))) APINT_TYPE_PREFIXint64;
typedef unsigned int __attribute__ ((bitwidth(64))) APINT_TYPE_PREFIXuint64;
#    define bool APINT_TYPE_PREFIXuint1
#else
typedef int __attribute__ ((bitwidth(64))) int64;
typedef unsigned int __attribute__ ((bitwidth(64))) uint64;
#    define bool uint1
#endif 
#    define true 1
#    define false 0

#  endif 


#endif /*#ifdef _cplusplus & C_TEST */

#undef _TY_CONCAT

#endif

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
