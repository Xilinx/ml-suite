/* autopilot_apint.h*/
/*
 * __VIVADO_HLS_COPYRIGHT-INFO__ 
 *
 * $Id$
 */

#ifndef _AUTOPILOT_APINT_H_
#define _AUTOPILOT_APINT_H_

#include "etc/autopilot_dt.h"
#include "etc/autopilot_ssdm_bits.h"

#define APInt_BitWidthOf(X) _ssdm_op_bitwidthof(X)

#define APInt_Concatenate(X, Y) _ssdm_op_Concat(X, Y)
#define APInt_GetBit(Val, Bit) _ssdm_op_GetBit(Val, Bit)
#define APInt_SetBit(Val, Bit, Repl) _ssdm_op_SetBit(Val, Bit, Repl)
#define APInt_GetRange(Val, Hi, Lo) _ssdm_op_GetRange(Val, Hi, Lo)
#define APInt_SetRange(Val, Hi, Lo, Repl) _ssdm_op_SetRange(Val, Hi, Lo, Repl)

#define APInt_AndReduce(X) _ssdm_op_reduce(and, X)
#define APInt_NandReduce(X) _ssdm_op_reduce(nand, X)
#define APInt_OrReduce(X) _ssdm_op_reduce(or, X)
#define APInt_NorReduce(X) _ssdm_op_reduce(nor, X)
#define APInt_XorReduce(X) _ssdm_op_reduce(xor, X)
#define APInt_XnorReduce(X) _ssdm_op_reduce(nxor, X)

#define APInt_BitsFromString(Str, Bits) \
    _ssdm_op_bitsFromString(Str, Bits)
#define APInt_BitsFromHexString(Str, Bits) \
    _ssdm_op_bitsFromHexString(Str, Bits)
#define APInt_BitsFromOctalString(Str, Bits) \
    _ssdm_op_bitsFromOctalString(Str, Bits)
#define APInt_BitsFromBinaryString(Str, Bits) \
    _ssdm_op_bitsFromBinaryString(Str, Bits)


/************************************************/

#define apint_bitwidthof(X) _ssdm_op_bitwidthof(X)

#define apint_concatenate(X, Y) _ssdm_op_Concat(X, Y)
#define apint_get_bit(Val, Bit) _ssdm_op_GetBit(Val, Bit)
#define apint_set_bit(Val, Bit, Repl) _ssdm_op_SetBit(Val, Bit, Repl)
#define apint_get_range(Val, Hi, Lo) _ssdm_op_GetRange(Val, Hi, Lo)
#define apint_set_range(Val, Hi, Lo, Repl) _ssdm_op_SetRange(Val, Hi, Lo, Repl)

#define apint_and_reduce(X) _ssdm_op_reduce(and, X)
#define apint_nand_reduce(X) _ssdm_op_reduce(nand, X)
#define apint_or_reduce(X) _ssdm_op_reduce(or, X)
#define apint_nor_reduce(X) _ssdm_op_reduce(nor, X)
#define apint_xor_reduce(X) _ssdm_op_reduce(xor, X)
#define apint_xnor_reduce(X) _ssdm_op_reduce(nxor, X)

#define apint_print(Val, Radix) _ssdm_op_print_apint(Val, Radix)
#define apint_fprint(FileOut, Val, Radix) _ssdm_op_fprint_apint(FileOut, Val, Radix)

#define apint_vstring2bits_hex(Str, Bits) _ssdm_op_sscan_hex_apint(Str, Bits)
#define apint_string2bits_dec(Str, Bits) _ssdm_op_bitsFromString(Str, Bits)
#define apint_string2bits_hex(Str, Bits) _ssdm_op_bitsFromHexString(Str, Bits)
#define apint_string2bits_oct(Str, Bits) _ssdm_op_bitsFromOctalString(Str, Bits)
#define apint_string2bits_bin(Str, Bits) _ssdm_op_bitsFromBinaryString(Str, Bits)
#define apint_string2bits(Str, Bits) apint_string2bits_dec(Str, Bits)

#endif

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
