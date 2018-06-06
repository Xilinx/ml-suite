/* -*- c++ -*-*/
/*
 * __VIVADO_HLS_COPYRIGHT-INFO__ 
 *
 *
 */

#ifndef X_HLS_FFT_H
#define X_HLS_FFT_H

/*
 * This file contains a C++ model of hls::fft.
 * It defines Vivado_HLS synthesis model.
 */
#ifndef __cplusplus
#error C++ is required to include this header file
#else


#include "ap_int.h"
#include <complex>
#include <math.h>

#ifndef AESL_SYN
#include <iostream>
#include "hls_stream.h"
#include "fft/xfft_v9_0_bitacc_cmodel.h"
#endif

namespace hls {

#ifdef AESL_SYN
#include "etc/autopilot_ssdm_op.h"
#endif

namespace ip_fft {

#ifndef INLINE
#define INLINE inline __attribute__((always_inline))
#endif

static const char* fftErrChkHead = "ERROR:hls::fft ";

enum ordering {bit_reversed_order = 0, natural_order};
enum scaling {scaled = 0, unscaled, block_floating_point};
enum arch {
    radix_4_burst_io = 1, radix_2_burst_io,
    pipelined_streaming_io, radix_2_lite_burst_io
};
enum rounding {truncation = 0, convergent_rounding};
enum mem { block_ram = 0, distributed_ram };
enum opt { 
    use_luts = 0, use_mults_resources, 
    use_mults_performance, use_xtremedsp_slices
};
enum type { fixed_point = 0, floating_point };
static const char* fft_data_format_str[] = {"fixed_point", "floating_point"};

struct params_t 
{
    static const unsigned input_width = 16;
    static const unsigned output_width = 16;
    static const unsigned status_width = 8;
    static const unsigned config_width = 16;
    static const unsigned max_nfft = 10;

    static const bool has_nfft = false; 
    static const unsigned  channels = 1;
    static const unsigned arch_opt = pipelined_streaming_io;
    static const unsigned phase_factor_width = 16;
    static const unsigned ordering_opt = bit_reversed_order;
    static const bool ovflo = true;
    static const unsigned scaling_opt = scaled;
    static const unsigned rounding_opt = truncation;
    static const unsigned mem_data = block_ram;
    static const unsigned mem_phase_factors = block_ram;
    static const unsigned mem_reorder = block_ram;
    static const unsigned stages_block_ram = (max_nfft < 10) ? 0 : (max_nfft - 9);
    static const bool mem_hybrid = false;
    static const unsigned complex_mult_type = use_mults_resources;
    static const unsigned butterfly_type = use_luts;

//not supported params:
    static const bool xk_index = false;
    static const bool cyclic_prefix_insertion = false; 
};

template <typename CONFIG_T>
struct config_t 
{
    config_t() { 

    }    

    ap_uint<CONFIG_T::config_width> data;
    // Check CONFIG_T::config_width
    INLINE void checkBitWidth(ip_fft::type data_type = ip_fft::fixed_point)
    {
    #ifndef AESL_SYN
        const unsigned max_nfft = CONFIG_T::max_nfft; 
        const unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        const unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        const unsigned ch_bits = CONFIG_T::channels;
        const unsigned arch = CONFIG_T::arch_opt;
        const unsigned tmp_bits = (arch == unsigned(ip_fft::pipelined_streaming_io) || arch == unsigned(ip_fft::radix_4_burst_io)) ? ((max_nfft+1)>>1) * 2 : 2 * max_nfft;
        //Temporarily set floating point type to always generate scaling due to bugs in FFT IP
        const bool need_scaling = (data_type == ip_fft::floating_point) ? true : (CONFIG_T::scaling_opt == unsigned(ip_fft::scaled));
        const unsigned sch_bits = need_scaling ? tmp_bits : 0;
        const unsigned config_bits = (sch_bits + ch_bits) * CONFIG_T::channels + cp_len_bits + nfft_bits;
        const unsigned config_width = ((config_bits + 7) >> 3) << 3; // padding
        if (CONFIG_T::config_width != config_width)
        {
            std::cerr << ip_fft::fftErrChkHead << "Config channel width = " << (int)CONFIG_T::config_width
                      << " is illegal." << std::endl;
            std::cerr << "Correct width is " << config_width << ". Please refer to FFT IP in Vivado GUI for details" << std::endl;
            exit(1);
        }
    #endif
    }


    INLINE void checkNfft(bool has_nfft)
    {
    #ifndef AESL_SYN
        if (has_nfft == 0)
        {
            std::cerr << fftErrChkHead << "FFT_HAS_NFFT = false."
                      << " It's invalid to access NFFT field."
                      << std::endl;
            exit(1);          
        }
    #endif
    }

    INLINE void checkCpLen(bool cp_len_enable)
    {
    #ifndef AESL_SYN
        if (cp_len_enable == 0)
        {
            std::cerr << fftErrChkHead << "FFT_CYCLIC_PREFIX_INSERTION = false."
                      << " It's invalid to access cp_len field."
                      << std::endl;
            exit(1);          
        }
    #endif
    }

    INLINE void checkSch(unsigned scaling_opt)
    {
    #ifndef AESL_SYN
        if (scaling_opt != unsigned(scaled))
        {
            std::cerr << fftErrChkHead << "FFT_SCALING != scaled."
                      << " It's invalid to access scaling_sch field."
                      << std::endl;
            exit(1);          
        }
    #endif
    }

    INLINE void setNfft(unsigned nfft)
    {
        //checkBitWidth();
        checkNfft(CONFIG_T::has_nfft);
        data.range(7, 0) = nfft;
    }
    INLINE unsigned getNfft()
    {
        //checkBitWidth();
        checkNfft(CONFIG_T::has_nfft);
        return data.range(7, 0);
    }
    INLINE unsigned getNfft() const
    {
        //checkBitWidth();
        checkNfft(CONFIG_T::has_nfft);
        return data.range(7, 0);
    }

    INLINE void setCpLen(unsigned cp_len) 
    {
        //checkBitWidth();
        checkCpLen(CONFIG_T::cyclic_prefix_insertion);
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        data.range(cp_len_bits+nfft_bits-1, nfft_bits) = cp_len;
    }
    INLINE unsigned getCpLen()
    {
        //checkBitWidth();
        checkCpLen(CONFIG_T::cyclic_prefix_insertion);
        unsigned ret = 0;
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        ret = data.range(cp_len_bits+nfft_bits-1, nfft_bits);
        return 0;
    }
    INLINE unsigned getCpLen() const
    {
        //checkBitWidth();
        checkCpLen(CONFIG_T::cyclic_prefix_insertion);
        unsigned ret = 0;
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        ret = data.range(cp_len_bits+nfft_bits-1, nfft_bits);
        return 0;
    }

    INLINE void setDir(bool dir, unsigned ch = 0)
    {
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        unsigned ch_lo = cp_len_bits + nfft_bits;
        unsigned ch_bits = 1;
        data.range(ch_bits*(ch+1)+ch_lo-1, ch_bits*ch+ch_lo) = dir;
    }
    INLINE unsigned getDir(unsigned ch = 0)
    { 
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        unsigned ch_lo = cp_len_bits + nfft_bits;
        unsigned ch_bits = 1;
        return data.range(ch_bits*(ch+1)+ch_lo-1, ch_bits*ch+ch_lo);
    }
    INLINE unsigned getDir(unsigned ch = 0) const
    { 
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        unsigned ch_lo = cp_len_bits + nfft_bits;
        unsigned ch_bits = 1;
        return data.range(ch_bits*(ch+1)+ch_lo-1, ch_bits*ch+ch_lo);
    }

    INLINE void setSch(unsigned sch, unsigned ch = 0)
    {
        //checkBitWidth();
        checkSch(CONFIG_T::scaling_opt);
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        unsigned ch_lo = cp_len_bits + nfft_bits;
        unsigned ch_bits = 1;
        unsigned arch = CONFIG_T::arch_opt;
        unsigned tmp_bits = (arch == unsigned(pipelined_streaming_io) || arch == unsigned(radix_4_burst_io)) ? ((max_nfft+1)>>1) * 2 : 2 * max_nfft;
        unsigned sch_bits = (CONFIG_T::scaling_opt == unsigned(scaled)) ? tmp_bits : 0;
        unsigned sch_lo = ch_lo + CONFIG_T::channels * ch_bits;
        data.range(sch_bits*(ch+1)+sch_lo-1, sch_bits*ch+sch_lo) = sch;
    }
    INLINE unsigned getSch(unsigned ch = 0)
    {
        //checkBitWidth();
        checkSch(CONFIG_T::scaling_opt);
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        unsigned ch_lo = cp_len_bits + nfft_bits;
        unsigned ch_bits = 1;
        unsigned arch = CONFIG_T::arch_opt;
        unsigned tmp_bits = (arch == unsigned(pipelined_streaming_io) || arch == unsigned(radix_4_burst_io)) ? ((max_nfft+1)>>1) * 2 : 2 * max_nfft;
        unsigned sch_bits = (CONFIG_T::scaling_opt == unsigned(scaled)) ? tmp_bits : 0;
        unsigned sch_lo = ch_lo + CONFIG_T::channels * ch_bits;
        return data.range(sch_bits*(ch+1)+sch_lo-1, sch_bits*ch+sch_lo);
    }
    INLINE unsigned getSch(unsigned ch = 0) const
    {
        //checkBitWidth();
        checkSch(CONFIG_T::scaling_opt);
        unsigned max_nfft = CONFIG_T::max_nfft; 
        unsigned nfft_bits = CONFIG_T::has_nfft ? 8 : 0; // Padding to 8 bits
        unsigned cp_len_bits = CONFIG_T::cyclic_prefix_insertion ? (((max_nfft + 7) >> 3) << 3) : 0; // Padding
        unsigned ch_lo = cp_len_bits + nfft_bits;
        unsigned ch_bits = 1;
        unsigned arch = CONFIG_T::arch_opt;
        unsigned tmp_bits = (arch == unsigned(pipelined_streaming_io) || arch == unsigned(radix_4_burst_io)) ? ((max_nfft+1)>>1) * 2 : 2 * max_nfft;
        unsigned sch_bits = (CONFIG_T::scaling_opt == unsigned(scaled)) ? tmp_bits : 0;
        unsigned sch_lo = ch_lo + CONFIG_T::channels * ch_bits;
        return data.range(sch_bits*(ch+1)+sch_lo-1, sch_bits*ch+sch_lo);
    }
};

template<typename CONFIG_T>
struct status_t 
{
    typedef ap_uint<CONFIG_T::status_width> status_data_t;
    status_data_t data;


    // Check CONFIG_T::status_width
    INLINE void checkBitWidth()
    {
    #ifndef AESL_SYN
        const bool has_ovflo = CONFIG_T::ovflo && (CONFIG_T::scaling_opt == unsigned(ip_fft::scaled));
        const unsigned blk_exp_bits = (CONFIG_T::scaling_opt == unsigned(ip_fft::block_floating_point)) ? 8 : 0; // padding to 8 bits
        const unsigned ovflo_bits = has_ovflo ? 1 : 0; 
        const unsigned status_bits = (blk_exp_bits + ovflo_bits) * CONFIG_T::channels;
        const unsigned status_width = (status_bits == 0) ? 8 : ((status_bits + 7) >> 3) << 3; // padding
        if (CONFIG_T::status_width != status_width)
        {
            std::cerr << ip_fft::fftErrChkHead << "Status channel width = " << (int)CONFIG_T::status_width
                      << " is illegal." << std::endl;
            exit(1);
        }
    #endif
    }

    INLINE void checkBlkExp(unsigned scaling_opt)
    {
    #ifndef AESL_SYN
        if (scaling_opt != unsigned(block_floating_point))
        {
            std::cerr << fftErrChkHead << "FFT_SCALING != block_floating_point."
                      << " It's invalid to access BLK_EXP field."
                      << std::endl;
            exit(1);          
        }
    #endif
    }

    INLINE void checkOvflo(bool has_ovflo)
    {
    #ifndef AESL_SYN
        if (!has_ovflo)
        {
            std::cerr << fftErrChkHead
                      << "Current configuration disables over flow field,"
                      << " it's invalid to access OVFLO field."
                      << std::endl;
            exit(1);          
        }
    #endif
    }

    INLINE void setBlkExp(status_data_t exp)
    {
        checkBitWidth();
        checkBlkExp(CONFIG_T::scaling_opt); 
        data = exp;
    }
    INLINE unsigned getBlkExp(unsigned ch = 0)
    {
        checkBitWidth();
        unsigned blk_exp_bits = (CONFIG_T::scaling_opt == unsigned(block_floating_point)) ? 8 : 0; // padding to 8 bits
        checkBlkExp(CONFIG_T::scaling_opt); 
        return data.range(blk_exp_bits*(ch+1)-1, blk_exp_bits*ch); 
    }
    INLINE unsigned getBlkExp(unsigned ch = 0) const
    {
        checkBitWidth();
        unsigned blk_exp_bits = (CONFIG_T::scaling_opt == unsigned(block_floating_point)) ? 8 : 0; // padding to 8 bits
        checkBlkExp(CONFIG_T::scaling_opt); 
        return data.range(blk_exp_bits*(ch+1)-1, blk_exp_bits*ch); 
    }

    INLINE void setOvflo(status_data_t ovflo)
    {
        checkBitWidth();
        bool has_ovflo = CONFIG_T::ovflo && (CONFIG_T::scaling_opt == unsigned(scaled));
        checkOvflo(has_ovflo);
        data = ovflo;
    }
    INLINE unsigned getOvflo(unsigned ch = 0)
    {
        checkBitWidth();
        bool has_ovflo = CONFIG_T::ovflo && (CONFIG_T::scaling_opt == unsigned(scaled));
        unsigned ovflo_bits = has_ovflo ? 1 : 0; 
        checkOvflo(has_ovflo);
        return data.range(ovflo_bits*(ch+1)-1, ovflo_bits*ch);
    }
    INLINE unsigned getOvflo(unsigned ch = 0) const
    {
        checkBitWidth();
        bool has_ovflo = CONFIG_T::ovflo && (CONFIG_T::scaling_opt == unsigned(scaled));
        unsigned ovflo_bits = has_ovflo ? 1 : 0; 
        checkOvflo(has_ovflo);
        return data.range(ovflo_bits*(ch+1)-1, ovflo_bits*ch);
    }
};

} // namespace hls::ip_fft

using namespace std;

template<
    typename CONFIG_T,
    char FFT_INPUT_WIDTH,
    char FFT_OUTPUT_WIDTH,
    typename FFT_INPUT_T,
    typename FFT_OUTPUT_T,
    int FFT_LENGTH,
    char FFT_CHANNELS,
    ip_fft::type FFT_DATA_FORMAT
>
INLINE void fft_core(
    complex<FFT_INPUT_T> xn[FFT_CHANNELS][FFT_LENGTH],
    complex<FFT_OUTPUT_T> xk[FFT_CHANNELS][FFT_LENGTH],
    ip_fft::status_t<CONFIG_T>* status,
    ip_fft::config_t<CONFIG_T>* config_ch)
{
#ifdef AESL_SYN

//////////////////////////////////////////////
// C level synthesis models for hls::fft
//////////////////////////////////////////////
#pragma HLS inline

    _ssdm_op_SpecKeepValue(
        //"component_name", "xfft_0",
        "channels", FFT_CHANNELS,
        "transform_length", 1 << CONFIG_T::max_nfft,
        "implementation_options", CONFIG_T::arch_opt-1,
        "run_time_configurable_transform_length", CONFIG_T::has_nfft,
        "data_format", ip_fft::fft_data_format_str[FFT_DATA_FORMAT],
        "input_width", FFT_INPUT_WIDTH,
        "output_width", FFT_OUTPUT_WIDTH,
        "phase_factor_width", CONFIG_T::phase_factor_width,
        "scaling_options", CONFIG_T::scaling_opt,
        "rounding_modes", CONFIG_T::rounding_opt,
        "aclken", "true",
        "aresetn", "true",
        "ovflo", CONFIG_T::ovflo,
        "xk_index", CONFIG_T::xk_index,
        "throttle_scheme", "nonrealtime",
        "output_ordering", CONFIG_T::ordering_opt,
        "cyclic_prefix_insertion", CONFIG_T::cyclic_prefix_insertion,
        "memory_options_data", CONFIG_T::mem_data,
        "memory_options_phase_factors", CONFIG_T::mem_phase_factors,
        "memory_options_reorder", CONFIG_T::mem_reorder,
        "number_of_stages_using_block_ram_for_data_and_phase_factors", CONFIG_T::stages_block_ram,
        "memory_options_hybrid", CONFIG_T::mem_hybrid,
        "complex_mult_type", CONFIG_T::complex_mult_type,
        "butterfly_type", CONFIG_T::butterfly_type
    );


    bool has_scaling_sch =  config_ch->getSch();
    bool has_direction = config_ch->getDir();

    if ( has_direction || has_scaling_sch )
        for (int i = 0; i < FFT_LENGTH; ++i)
        {
            for (int c = 0; c < FFT_CHANNELS; ++c)
            {
            #pragma HLS unroll complete
                xk[c][i] = xn[c][i];
            }
        }

    status->data = config_ch->getDir();

#else

//////////////////////////////////////////////
// C level simulation models for hls::fft
//////////////////////////////////////////////

    // Declare the C model IO structures
    xilinx_ip_xfft_v9_0_generics  generics;
    xilinx_ip_xfft_v9_0_state    *state;
    xilinx_ip_xfft_v9_0_inputs    inputs;
    xilinx_ip_xfft_v9_0_outputs   outputs;

    // Log2 of FFT length
    int fft_length = FFT_LENGTH;
    int NFFT = 0;
    if (CONFIG_T::has_nfft)
        NFFT = config_ch->getNfft();
    else
        NFFT = CONFIG_T::max_nfft;

    const int samples =  1 << NFFT;

    ///////////// IP parameters legality checking /////////////

    // Check CONFIG_T::config_width
    config_ch->checkBitWidth(FFT_DATA_FORMAT);

    // Check CONFIG_T::status_width
    status->checkBitWidth();

    // Check ip parameters
    if (CONFIG_T::channels < 1 || CONFIG_T::channels > 12)
    {
        std::cerr << ip_fft::fftErrChkHead << "Channels = " << (int)CONFIG_T::channels
                  << " is illegal. It should be from 1 to 12."
                  << std::endl;
        exit(1);
    }

    if (CONFIG_T::max_nfft < 3 || CONFIG_T::max_nfft > 16)
    {
        std::cerr << ip_fft::fftErrChkHead << "NFFT_MAX = " << (int)CONFIG_T::max_nfft 
                  << " is illegal. It should be from 3 to 16."
                  << std::endl;
        exit(1);
    }

    unsigned length = FFT_LENGTH;
    if (!CONFIG_T::has_nfft)
    {
        if (FFT_LENGTH != (1 << CONFIG_T::max_nfft))
        {
            std::cerr << ip_fft::fftErrChkHead << "FFT_LENGTH = " << (int)FFT_LENGTH
                      << " is illegal. Log2(FFT_LENGTH) should equal to NFFT_MAX when run-time configurable length is disabled."
                      << std::endl;
            exit(1);
        }
    }
    else if (length & (length - 1))
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_LENGTH = " << (int)FFT_LENGTH
                  << " is illegal. It should be the integer power of 2."
                  << std::endl;
        exit(1);
    }
    else if (NFFT < 3 || NFFT > 16)
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_LENGTH = " << (int)FFT_LENGTH
                  << " is illegal. Log2(FFT_LENGTH) should be from 3 to 16."
                  << std::endl;
        exit(1);
    }
    else if (NFFT > CONFIG_T::max_nfft)
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_LENGTH = " << (int)FFT_LENGTH
                  << " is illegal. Log2(FFT_LENGTH) should be less than or equal to NFFT_MAX."
                  << std::endl;
        exit(1);
    } 
#if 0
    else if (NFFT != config_ch->getNfft())
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_LENGTH = " << (int)FFT_LENGTH
                  << " is illegal. Log2(FFT_LENGTH) should equal to NFFT field of configure channel."
                  << std::endl;
        exit(1);
    }
#endif

    if ((FFT_INPUT_WIDTH < 8) || (FFT_INPUT_WIDTH > 40))
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_INPUT_WIDTH = " << (int)FFT_INPUT_WIDTH
                  << " is illegal. It should be 8,16,24,32,40."
                  << std::endl;
        exit(1);
    }

    if (CONFIG_T::scaling_opt == ip_fft::unscaled && FFT_DATA_FORMAT != ip_fft::floating_point)
    {
        unsigned golden = FFT_INPUT_WIDTH + CONFIG_T::max_nfft + 1;
        golden = ((golden + 7) >> 3) << 3;
        if (FFT_OUTPUT_WIDTH != golden)
        {
            std::cerr << ip_fft::fftErrChkHead << "FFT_OUTPUT_WIDTH = " << (int)FFT_OUTPUT_WIDTH
                      << " is illegal with unscaled arithmetic. It should be input_width+nfft_max+1."
                      << std::endl;
            exit(1);
        }
    }
    else if (FFT_OUTPUT_WIDTH != FFT_INPUT_WIDTH)
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_OUTPUT_WIDTH = " << (int)FFT_OUTPUT_WIDTH
                  << " is illegal. It should be the same as input_width."
                  << std::endl;
        exit(1);
    }

    if (CONFIG_T::channels > 1 && CONFIG_T::arch_opt == ip_fft::pipelined_streaming_io)
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_CHANNELS = " << (int)CONFIG_T::channels << " and FFT_ARCH = pipelined_streaming_io"
                  << " is illegal. pipelined_streaming_io architecture is not supported when channels is bigger than 1."
                  << std::endl;
        exit(1);
    }

    if (CONFIG_T::channels > 1 && FFT_DATA_FORMAT == ip_fft::floating_point)
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_CHANNELS = " << (int)CONFIG_T::channels
                  << " is illegal with floating point data format. Floating point data format only supports 1 channel."
                  << std::endl;
        exit(1);
    }

    if (FFT_DATA_FORMAT == ip_fft::floating_point)
    {
        if (CONFIG_T::phase_factor_width != 24 && CONFIG_T::phase_factor_width != 25)
        {
            std::cerr << ip_fft::fftErrChkHead << "FFT_PHASE_FACTOR_WIDTH = " << (int)CONFIG_T::phase_factor_width
                      << " is illegal with floating point data format. It should be 24 or 25."
                      << std::endl;
            exit(1);
        }
    } 
    else if (CONFIG_T::phase_factor_width < 8 || CONFIG_T::phase_factor_width > 34)
    {
        std::cerr << ip_fft::fftErrChkHead << "FFT_PHASE_FACTOR_WIDTH = " << (int)CONFIG_T::phase_factor_width
                  << " is illegal. It should be from 8 to 34."
                  << std::endl;
        exit(1);
    }

    //////////////////////////////////////////////////////////

    // Build up the C model generics structure
    generics.C_NFFT_MAX      = CONFIG_T::max_nfft;
    generics.C_ARCH          = CONFIG_T::arch_opt;
    generics.C_HAS_NFFT      = CONFIG_T::has_nfft;
    generics.C_INPUT_WIDTH   = FFT_INPUT_WIDTH;
    generics.C_TWIDDLE_WIDTH = CONFIG_T::phase_factor_width;
    generics.C_HAS_SCALING   = CONFIG_T::scaling_opt == ip_fft::unscaled ? 0 : 1; 
    generics.C_HAS_BFP       = CONFIG_T::scaling_opt == ip_fft::block_floating_point ? 1 : 0;
    generics.C_HAS_ROUNDING  = CONFIG_T::rounding_opt;
    generics.C_USE_FLT_PT    = FFT_DATA_FORMAT == ip_fft::floating_point ? 1 : 0;

    // Create an FFT state object
    state = xilinx_ip_xfft_v9_0_create_state(generics);

    int stages = 0;
    if ((generics.C_ARCH == 2) || (generics.C_ARCH == 4))  // radix-2
        stages = NFFT;
    else  // radix-4 or radix-22
        stages = (NFFT+1)/2;

    double* xn_re       = (double*) malloc(samples * sizeof(double));
    double* xn_im       = (double*) malloc(samples * sizeof(double));
    int*    scaling_sch = (int*)    malloc(stages  * sizeof(int));
    double* xk_re       = (double*) malloc(samples * sizeof(double));
    double* xk_im       = (double*) malloc(samples * sizeof(double));

    // Check the memory was allocated successfully for all arrays
    if (xn_re == NULL || xn_im == NULL || scaling_sch == NULL || xk_re == NULL || xk_im == NULL)
    {
      std::cerr << "Couldn't allocate memory for input and output data arrays - dying" << std::endl;
      exit(3);
    }

    ap_uint<CONFIG_T::status_width> overflow = 0;
    ap_uint<CONFIG_T::status_width> blkexp = 0;
    for (int c = 0; c < FFT_CHANNELS; ++c)
    {
        // Set pointers in input and output structures
        inputs.xn_re       = xn_re;
        inputs.xn_im       = xn_im;
        inputs.scaling_sch = scaling_sch;
        outputs.xk_re      = xk_re;
        outputs.xk_im      = xk_im;

        // Store in inputs structure
        inputs.nfft = NFFT;
        // config data
        inputs.direction = config_ch->getDir(c);
        unsigned scaling = 0;
        if (CONFIG_T::scaling_opt == ip_fft::scaled) 
            scaling = config_ch->getSch(c);
        for (int i = 0; i < stages; i++)
        {
            inputs.scaling_sch[i] = scaling & 0x3;
            scaling >>= 2;
        }
        inputs.scaling_sch_size = stages;
        for (int i = 0; i < samples ; i++)
        {
            complex<FFT_INPUT_T> din = xn[c][i];
            inputs.xn_re[i] = (double)din.real();
            inputs.xn_im[i] = (double)din.imag();
            #ifdef DEBUG
            std::cout << "xn[" << c "][" << i << ": xn_re = " << inputs .xn_re[i] << 
                    " xk_im = " <<  inputs.xn_im[i] << endl;
            #endif
        }
        inputs.xn_re_size = samples;
        inputs.xn_im_size = samples;

        // Set sizes of output structure arrays
        outputs.xk_re_size    = samples;
        outputs.xk_im_size    = samples;

        //#define DEBUG
        #ifdef DEBUG
        ///////////////////////////////////////////////////////////////////////////////
        /// Debug
        std::cout << "About to call the C model with:" << std::endl;
        std::cout << "Generics:" << std::endl;
        std::cout << "  C_NFFT_MAX = "      << generics.C_NFFT_MAX << std::endl;
        std::cout << "  C_ARCH = "          << generics.C_ARCH << std::endl;
        std::cout << "  C_HAS_NFFT = "      << generics.C_HAS_NFFT << std::endl;
        std::cout << "  C_INPUT_WIDTH = "   << generics.C_INPUT_WIDTH << std::endl;
        std::cout << "  C_TWIDDLE_WIDTH = " << generics.C_TWIDDLE_WIDTH << std::endl;
        std::cout << "  C_HAS_SCALING = "   << generics.C_HAS_SCALING << std::endl;
        std::cout << "  C_HAS_BFP = "       << generics.C_HAS_BFP << std::endl;
        std::cout << "  C_HAS_ROUNDING = "  << generics.C_HAS_ROUNDING << std::endl;
        std::cout << "  C_USE_FLT_PT = "    << generics.C_USE_FLT_PT << std::endl;
        
        std::cout << "Inputs structure:" << std::endl;
        std::cout << "  nfft = " << inputs.nfft << std::endl;
        printf("  xn_re[0] = %e\n",inputs.xn_re[0]);
        std::cout << "  xn_re_size = " << inputs.xn_re_size << std::endl;
        printf("  xn_im[0] = %e\n",inputs.xn_im[0]);
        std::cout << "  xn_im_size = " << inputs.xn_im_size << std::endl;

        for (int i = stages - 1; i >= 0; --i)
            std::cout << "  scaling_sch[" << i << "] = " << inputs.scaling_sch[i] << std::endl;

        std::cout << "  scaling_sch_size = " << inputs.scaling_sch_size << std::endl;
        std::cout << "  direction = " << inputs.direction << std::endl;
        
        std::cout << "Outputs structure:" << std::endl;
        std::cout << "  xk_re_size = " << outputs.xk_re_size << std::endl;
        std::cout << "  xk_im_size = " << outputs.xk_im_size << std::endl;
                
        // Run the C model to generate output data
        std::cout << "Running the C model..." << std::endl;
        ///////////////////////////////////////////////////////////////////////////////
        #endif

        int result = 0;
        result = xilinx_ip_xfft_v9_0_bitacc_simulate(state, inputs, &outputs);
        if (result != 0)
        {
          std::cerr << "An error occurred when simulating the FFT core: return code " << result << std::endl;
          exit(4);
        }

        // Output data
        for (int i = 0; i < samples; i++)
        {
            complex<FFT_OUTPUT_T> dout;
            unsigned addr_reverse = 0;
            for (int k = 0; k < NFFT; ++k)
            {
                addr_reverse <<= 1;
                addr_reverse |= (i >> k) & 0x1;
            }
            unsigned addr = i;
            if (CONFIG_T::ordering_opt == ip_fft::bit_reversed_order)
                addr = addr_reverse;
            dout = complex<FFT_OUTPUT_T> (outputs.xk_re[addr], outputs.xk_im[addr]);
            xk[c][i] = dout;
            #ifdef DEBUG
            cout << "xk[" << c "][" << i << ": xk_re = " << outputs.xk_re[addr] << 
                    " xk_im = " <<  outputs.xk_im[addr] << endl;
            #endif
        }
        
        // Status
        if (CONFIG_T::scaling_opt == ip_fft::block_floating_point)
            blkexp.range(c*8+7, c*8) = outputs.blk_exp;
        else if (CONFIG_T::ovflo && (CONFIG_T::scaling_opt == ip_fft::scaled))
           overflow.range(c, c) = outputs.overflow; 
    }

    // Status
    if (CONFIG_T::scaling_opt == ip_fft::block_floating_point)
        status->setBlkExp(blkexp);
    else if (CONFIG_T::ovflo && (CONFIG_T::scaling_opt == ip_fft::scaled))
        status->setOvflo(overflow);

    // Release memory used for input and output arrays
    free(xn_re);
    free(xn_im);
    free(scaling_sch);
    free(xk_re);
    free(xk_im);

    // Destroy FFT state to free up memory
    xilinx_ip_xfft_v9_0_destroy_state(state);
#endif

} // End of fft_core


template<
    typename CONFIG_T,
    char FFT_INPUT_WIDTH,
    char FFT_OUTPUT_WIDTH,
    typename FFT_INPUT_T,
    typename FFT_OUTPUT_T,
    int FFT_LENGTH,
    char FFT_CHANNELS,
    ip_fft::type FFT_DATA_FORMAT
>
INLINE void fft_core(
    complex<FFT_INPUT_T> xn[FFT_LENGTH],
    complex<FFT_OUTPUT_T> xk[FFT_LENGTH],
    ip_fft::status_t<CONFIG_T>* status,
    ip_fft::config_t<CONFIG_T>* config_ch)
{
#ifdef AESL_SYN
#pragma HLS inline

    _ssdm_op_SpecKeepValue(
        //"component_name", "xfft_0",
        "channels", FFT_CHANNELS,
        "transform_length", FFT_LENGTH,
        "implementation_options", CONFIG_T::arch_opt-1,
        "run_time_configurable_transform_length", CONFIG_T::has_nfft,
        "data_format", ip_fft::fft_data_format_str[FFT_DATA_FORMAT],
        "input_width", FFT_INPUT_WIDTH,
        "output_width", FFT_OUTPUT_WIDTH,
        "phase_factor_width", CONFIG_T::phase_factor_width,
        "scaling_options", CONFIG_T::scaling_opt,
        "rounding_modes", CONFIG_T::rounding_opt,
        "aclken", "true",
        "aresetn", "true",
        "ovflo", CONFIG_T::ovflo,
        "xk_index", CONFIG_T::xk_index,
        "throttle_scheme", "nonrealtime",
        "output_ordering", CONFIG_T::ordering_opt,
        "cyclic_prefix_insertion", CONFIG_T::cyclic_prefix_insertion,
        "memory_options_data", CONFIG_T::mem_data,
        "memory_options_phase_factors", CONFIG_T::mem_phase_factors,
        "memory_options_reorder", CONFIG_T::mem_reorder,
        "number_of_stages_using_block_ram_for_data_and_phase_factors", CONFIG_T::stages_block_ram,
        "memory_options_hybrid", CONFIG_T::mem_hybrid,
        "complex_mult_type", CONFIG_T::complex_mult_type,
        "butterfly_type", CONFIG_T::butterfly_type
    );


    bool has_scaling_sch =  config_ch->getSch();
    bool has_direction = config_ch->getDir();

    if ( has_direction || has_scaling_sch )
        for (int i = 0; i < FFT_LENGTH; ++i)
        {
            xk[i] = xn[i]; 
        }

    status->data = config_ch->getDir();

#else
    complex<FFT_INPUT_T> xn_multi_chan [1][FFT_LENGTH];
    complex<FFT_OUTPUT_T> xk_multi_chan [1][FFT_LENGTH];

    for(int i=0; i< FFT_LENGTH; i++)
        xn_multi_chan[0][i] = xn[i];

    fft_core<
        CONFIG_T,
        FFT_INPUT_WIDTH,
        FFT_OUTPUT_WIDTH,
        FFT_INPUT_T,
        FFT_OUTPUT_T,
        FFT_LENGTH,
        1,
        FFT_DATA_FORMAT
    >(xn_multi_chan, xk_multi_chan, status, config_ch);       

    for(int i=0; i< FFT_LENGTH; i++)
        xk[i] = xk_multi_chan[0][i];
#endif
}


// 1-channel, fixed-point
template<
    typename CONFIG_T
>
void fft(
    complex<ap_fixed<((CONFIG_T::input_width+7)/8)*8, 1> > xn[1 << CONFIG_T::max_nfft],
    complex<ap_fixed<((CONFIG_T::output_width+7)/8)*8, ((CONFIG_T::output_width+7)/8)*8-CONFIG_T::input_width+1> > xk[1 << CONFIG_T::max_nfft],
    ip_fft::status_t<CONFIG_T>* status,
    ip_fft::config_t<CONFIG_T>* config_ch)
{
#pragma HLS inline off 
#pragma HLS resource core="Vivado_FFT" variable=return metadata="parameterizable"
//#pragma HLS function instantiate variable=core_params
#pragma HLS interface ap_fifo port=config_ch
#pragma HLS interface ap_fifo port=status
#pragma HLS interface ap_fifo port=xn
#pragma HLS interface ap_fifo port=xk

#pragma HLS data_pack variable=xn
#pragma HLS data_pack variable=xk

    fft_core<
        CONFIG_T,
        CONFIG_T::input_width,
        CONFIG_T::output_width,
        ap_fixed<((CONFIG_T::input_width+7)/8)*8, 1>,
        ap_fixed<((CONFIG_T::output_width+7)/8)*8, ((CONFIG_T::output_width+7)/8)*8-CONFIG_T::input_width+1>,
        1 << CONFIG_T::max_nfft,
        1,
        ip_fft::fixed_point
    >(xn, xk, status, config_ch);       

} // End of 1-channel, fixed-point

// Multi-channels, fixed-point
template<
    typename CONFIG_T
>
void fft(
    complex<ap_fixed<((CONFIG_T::input_width+7)/8)*8, 1> > xn[CONFIG_T::channels][1 << CONFIG_T::max_nfft],
    complex<ap_fixed<((CONFIG_T::output_width+7)/8)*8, 
                      ((CONFIG_T::output_width+7)/8)*8-CONFIG_T::input_width+1> > xk[CONFIG_T::channels][1 << CONFIG_T::max_nfft],
    ip_fft::status_t<CONFIG_T>* status,
    ip_fft::config_t<CONFIG_T>* config_ch)
{
#pragma HLS inline off 
#pragma HLS resource core="Vivado_FFT" variable=return metadata="parameterizable"
//#pragma HLS function instantiate variable=core_params
#pragma HLS interface ap_fifo port=config_ch
#pragma HLS interface ap_fifo port=status
#pragma HLS interface ap_fifo port=xn
#pragma HLS interface ap_fifo port=xk

#pragma HLS data_pack variable=xn
#pragma HLS data_pack variable=xk

//#if (FFT_CHANNELS > 1)
#pragma HLS array_reshape dim=1 variable=xn
#pragma HLS array_reshape dim=1 variable=xk
//#endif

        fft_core<
            CONFIG_T,
            CONFIG_T::input_width,
            CONFIG_T::output_width,
            ap_fixed<((CONFIG_T::input_width+7)/8)*8, 1>,
            ap_fixed<((CONFIG_T::output_width+7)/8)*8, ((CONFIG_T::output_width+7)/8)*8-CONFIG_T::input_width+1>,
            1 << CONFIG_T::max_nfft,
            CONFIG_T::channels,
            ip_fft::fixed_point
        >(xn, xk, status, config_ch);       
} // End of multi-channels, fixed-point

// 1-channel, floating-point
template<
    typename CONFIG_T
>
void fft(
    complex<float> xn[1 << CONFIG_T::max_nfft],
    complex<float> xk[1 << CONFIG_T::max_nfft],
    ip_fft::status_t<CONFIG_T>* status,
    ip_fft::config_t<CONFIG_T>* config_ch)
{
#pragma HLS inline off 
#pragma HLS resource core="Vivado_FFT" variable=return metadata="parameterizable"
//#pragma HLS function instantiate variable=core_params
#pragma HLS interface ap_fifo port=config_ch
#pragma HLS interface ap_fifo port=status
#pragma HLS interface ap_fifo port=xn
#pragma HLS interface ap_fifo port=xk
#pragma HLS data_pack variable=config_ch
#pragma HLS data_pack variable=xn
#pragma HLS data_pack variable=xk

    fft_core<
        CONFIG_T,
        32,
        32,
        float,
        float,
        1 << CONFIG_T::max_nfft,
        1,
        ip_fft::floating_point
    >(xn, xk, status, config_ch);       
} // End of 1-channel, floating-point

} // namespace hls
#endif // __cplusplus
#endif // X_HLS_FFT_H

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
