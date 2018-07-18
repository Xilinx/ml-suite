/* -*- c++ -*-*/
/*
#-  (c) Copyright 2011-2014 Xilinx, Inc. All rights reserved.
#-
#-  This file contains confidential and proprietary information
#-  of Xilinx, Inc. and is protected under U.S. and
#-  international copyright and other intellectual property
#-  laws.
#-
#-  DISCLAIMER
#-  This disclaimer is not a license and does not grant any
#-  rights to the materials distributed herewith. Except as
#-  otherwise provided in a valid license issued to you by
#-  Xilinx, and to the maximum extent permitted by applicable
#-  law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
#-  WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
#-  AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
#-  BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
#-  INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
#-  (2) Xilinx shall not be liable (whether in contract or tort,
#-  including negligence, or under any other theory of
#-  liability) for any loss or damage of any kind or nature
#-  related to, arising under or in connection with these
#-  materials, including for any direct, or any indirect,
#-  special, incidental, or consequential loss or damage
#-  (including loss of data, profits, goodwill, or any type of
#-  loss or damage suffered as a result of any action brought
#-  by a third party) even if such damage or loss was
#-  reasonably foreseeable or Xilinx had been advised of the
#-  possibility of the same.
#-
#-  CRITICAL APPLICATIONS
#-  Xilinx products are not designed or intended to be fail-
#-  safe, or for use in any application requiring fail-safe
#-  performance, such as life-support or safety devices or
#-  systems, Class III medical devices, nuclear facilities,
#-  applications related to the deployment of airbags, or any
#-  other applications that could lead to death, personal
#-  injury, or severe property or environmental damage
#-  (individually and collectively, "Critical
#-  Applications"). Customer assumes the sole risk and
#-  liability of any use of Xilinx products in Critical
#-  Applications, subject only to applicable laws and
#-  regulations governing limitations on product liability.
#-
#-  THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
#-  PART OF THIS FILE AT ALL TIMES. 
#- ************************************************************************

 *
 *
 */

#ifndef X_HLS_DDS_H
#define X_HLS_DDS_H

/*
 * This file contains a C++ model of hls::dds.
 * It defines Vivado_HLS synthesis model.
 */
#ifndef __cplusplus
#error C++ is required to include this header file
#else

#include "ap_int.h"
#include <limits.h>
#include <complex>
#include <math.h>

#ifndef AESL_SYN
#include <iostream>
#include "hls_stream.h"
//#include "dds/gmp.h" 
#include "dds/dds_compiler_v6_0_bitacc_cmodel.h"
#else

/**
 * Core Specific Constants
 */
/* PartsPresent values */
#define XIP_DDS_PHASE_GEN_AND_SIN_COS_LUT 0
#define XIP_DDS_PHASE_GEN_ONLY            1
#define XIP_DDS_SIN_COS_LUT_ONLY          2

  /* DDS_Clock_Rate limits */
#define XIP_DDS_CLOCK_RATE_MIN 0.01
#define XIP_DDS_CLOCK_RATE_MAX 1000

  /* Channels limits */
#define XIP_DDS_CHANNELS_MIN 1
#define XIP_DDS_CHANNELS_MAX 16

  /* Mode of Operation values */
#define XIP_DDS_MOO_CONVENTIONAL 0
#define XIP_DDS_MOO_RASTERIZED   1

  /* Modulus limits */
#define XIP_DDS_MODULUS_MIN 9
#define XIP_DDS_MODULUS_MAX 16384

  /* ParameterEntry values */
#define XIP_DDS_SYSTEM_PARAMS   0
#define XIP_DDS_HARDWARE_PARAMS 1

  /* Spurious Free Dynamic Range SFDR limits */
#define XIP_DDS_SFDR_MIN 18
#define XIP_DDS_SFDR_MAX 150

  /* Frequency_Resolution SFDR limits */
#define XIP_DDS_FREQ_RES_MIN 0.000000001
#define XIP_DDS_FREQ_RES_MAX 125000000

  /* Noise_Shaping values */
#define XIP_DDS_NS_NONE   0
#define XIP_DDS_NS_DITHER 1
#define XIP_DDS_NS_TAYLOR 2
#define XIP_DDS_NS_AUTO   3

  /* Phase_Increment and Phase_Offset values */
#define XIP_DDS_PINCPOFF_NONE   0
#define XIP_DDS_PINCPOFF_PROG   1
#define XIP_DDS_PINCPOFF_FIXED  2
#define XIP_DDS_PINCPOFF_STREAM 3

  /* Output_Selection values */
#define XIP_DDS_OUT_SIN_ONLY    0
#define XIP_DDS_OUT_COS_ONLY    1
#define XIP_DDS_OUT_SIN_AND_COS 2

  /* Present/absent values */
#define XIP_DDS_ABSENT 0
#define XIP_DDS_PRESENT 1

  /* Amplitude_Mode values */
#define XIP_DDS_FULL_RANGE  0
#define XIP_DDS_UNIT_CIRCLE 1

  /* Output_Form */
#define XIP_DDS_OUTPUT_TWOS     0
#define XIP_DDS_OUTPUT_SIGN_MAG 1

  /* Memory_Type values */
#define XIP_DDS_MEM_AUTO  0
#define XIP_DDS_MEM_BLOCK 1
#define XIP_DDS_MEM_DIST  2

  /* Optimization_Goal values */
#define XIP_DDS_OPTGOAL_AUTO  0
#define XIP_DDS_OPTGOAL_AREA  1
#define XIP_DDS_OPTGOAL_SPEED 2

  /* Resolved Optimization_Goal values */
#define XIP_DDS_RESOPTGOAL_AREA  0
#define XIP_DDS_RESOPTGOAL_SPEED 1

  /* DSP48_use values */
#define XIP_DDS_DSP_MIN 0
#define XIP_DDS_DSP_MAX 1

  /* Latency_configuration values */
#define XIP_DDS_LATENCY_AUTO   0
#define XIP_DDS_LATENCY_MANUAL 1

  /* S_CONFIG_Sync_Mode values */
#define XIP_DDS_CONFIG_SYNC_VECTOR 0
#define XIP_DDS_CONFIG_SYNC_PACKET 1

#define ci_max_pipe_stages 13
#define ci_quad_sym_thresh 2048

typedef double xip_dds_v6_0_data;
#endif

using namespace std;

namespace hls {

#ifdef AESL_SYN
#include "etc/autopilot_ssdm_op.h"
#endif

namespace ip_dds {

#ifndef INLINE
#define INLINE inline __attribute__((always_inline))
#endif

static const char* ddsErrChkHead = "ERROR:hls::dds ";

template<typename phase_t>
struct in_config_pinc{
     phase_t pinc;
};

template<typename phase_t>
struct in_config_poff{
     phase_t poff;
};

template<typename CONFIG_T>
struct in_config_pinc_poff{
static const unsigned __HLS_DDS_CONFIG_N_ = (CONFIG_T::Phase_Increment%2 + CONFIG_T::Phase_Offset%2);
static const unsigned input_axi_width = (CONFIG_T::Phase_Width%8) ? (CONFIG_T::Phase_Width/8+1)*8 : CONFIG_T::Phase_Width;    
static const unsigned output_axi_width = (CONFIG_T::Output_Width%8) ? (CONFIG_T::Output_Width/8+1)*8 : CONFIG_T::Output_Width;    

typedef ap_ufixed<input_axi_width, input_axi_width - CONFIG_T::phase_fractional_bits> out_phase_t;

    out_phase_t data[__HLS_DDS_CONFIG_N_];
    in_config_pinc_poff()
    {
     #pragma HLS array partition variable=data
    }
    in_config_pinc_poff(const in_config_pinc_poff &tmp)
    {
     #pragma HLS array partition variable=data
     data[0] = tmp.data[0];
if(__HLS_DDS_CONFIG_N_>1)
     data[1] = tmp.data[1];
    }

    inline void operator = (const in_config_pinc_poff &tmp)
    {
     data[0] = tmp.data[0];
if(__HLS_DDS_CONFIG_N_>1)
     data[1] = tmp.data[1];
    }

    void set_pinc(out_phase_t &tmp)
    {
#pragma HLS inline
#ifndef AESL_SYN
        if(CONFIG_T::Phase_Increment%2 == 0)
        {
            std::cout<<"error found"<<std::endl;
            assert(0&&"Phase_Increment is set to NONE or FIXED");
        }
#endif
        if(CONFIG_T::Phase_Increment%2 == 1)
            data[0] = tmp;
    }

    void set_poff(out_phase_t &tmp)
    {
#pragma HLS inline
#ifndef AESL_SYN
        if(CONFIG_T::Phase_Offset%2 == 0)
        {
            std::cout<<"error found"<<std::endl;
            assert(0&&"Phase_Offset is set to NONE or FIXED");
        }
#endif
        if(CONFIG_T::Phase_Offset%2 == 1)
            data[__HLS_DDS_CONFIG_N_-1] = tmp;
    }

    out_phase_t & get_pinc()
    {
#ifndef AESL_SYN
        if(CONFIG_T::Phase_Increment%2 == 0)
        {
            std::cout<<"error found"<<std::endl;
            assert(0&&"Phase_Increment is set to NONE or FIXED");
        }
#endif
        if(CONFIG_T::Phase_Increment%2 == 1)
            return data[0];
    }

    out_phase_t &get_poff()
    {
#ifndef AESL_SYN
        if(CONFIG_T::Phase_Offset%2 == 0)
        {
            std::cout<<"error found"<<std::endl;
            assert(0&&"Phase_Offset is set to NONE or FIXED");
        }
#endif
        if(CONFIG_T::Phase_Offset%2 == 1)
            return data[__HLS_DDS_CONFIG_N_-1];
    }

};

template<typename data_t>
struct out_data_sin{
     data_t sin;
};

template<typename data_t>
struct out_data_cos{
     data_t cos;
};

template<typename CONFIG_T>
struct out_data_sin_cos{
static const unsigned __HLS_DDS_OUT_N_ = (CONFIG_T::Output_Selection/2 + 1);
static const unsigned input_axi_width = (CONFIG_T::Phase_Width%8) ? (CONFIG_T::Phase_Width/8+1)*8 : CONFIG_T::Phase_Width;    
static const unsigned output_axi_width = (CONFIG_T::Output_Width%8) ? (CONFIG_T::Output_Width/8+1)*8 : CONFIG_T::Output_Width;    

typedef ap_uint<CONFIG_T::Phase_Width> config_t;
typedef ap_ufixed<input_axi_width, input_axi_width - CONFIG_T::phase_fractional_bits> out_phase_t;
typedef ap_fixed<output_axi_width, output_axi_width - CONFIG_T::output_fractional_bits>  out_data_t;

    out_data_t data[__HLS_DDS_OUT_N_];
    out_data_sin_cos()
    {
     #pragma HLS array partition variable=data
    }
    out_data_sin_cos(const out_data_sin_cos &tmp)
    {
     #pragma HLS array partition variable=data
     data[0] = tmp.data[0];
if(__HLS_DDS_OUT_N_>1)
     data[1] = tmp.data[1];
    }

    inline void operator = (const out_data_sin_cos &tmp)
    {
     data[0] = tmp.data[0];
if(__HLS_DDS_OUT_N_>1)
     data[1] = tmp.data[1];
    }

    out_data_t & get_sin()
    {
#ifndef AESL_SYN
        if(CONFIG_T::Output_Selection == 1)
        {
            std::cerr << ddsErrChkHead 
                      << "Please do NOT use get_sin() when set Output_Selection to XIP_DDS_OUT_COS_ONLY."
                      << std::endl;
            //assert(0&&"Output_Selection is set to COS_ONLY");
            exit(1);
        }
#endif
        if(CONFIG_T::Output_Selection%2 == 0)
            return data[__HLS_DDS_OUT_N_-1];
    }

    out_data_t &get_cos()
    {
#ifndef AESL_SYN
        if(CONFIG_T::Output_Selection == 0)
        {
            std::cerr << ddsErrChkHead 
                      << "Please do NOT use get_cos() when set Output_Selection to XIP_DDS_OUT_SIN_ONLY."
                      << std::endl;
            //assert(0&&"Output_Selection is set to SIN_ONLY");
            exit(1);
        }
#endif
        if(CONFIG_T::Output_Selection > 0)
            return data[0];
    }
};

enum filter_type {single_rate = 0, interpolation, decimation, hibert, interpolated};
static const char* dds_filter_type_str[] = {
    "single_rate", "interpolation", 
    "decimation", "hilbert", "interpolated"
};

enum rate_change_type {integer = 0, fixed_fractional};
static const char* dds_rate_change_type_str[] = {
    "integer", "fixed_fractional"
};

enum chan_seq {basic = 0, advanced};
static const char* dds_channel_sequence_str[] = {
    "basic", "advanced"
};

enum rate_specification {frequency = 0, period};
static const char* dds_ratespecification_str[] = {
    "frequency_specification", "sample_period"
};

enum value_sign {value_signed = 0, value_unsigned};
static const char* dds_value_sign_str[] = {"signed", "unsigned"};

enum quantization {integer_coefficients = 0, quantize_only, maximize_dynamic_range};
static const char* dds_quantization_str[] = {
    "integer_coefficients", "quantize_only", "maximize_dynamic_range"
};

enum coeff_structure {inferred = 0, non_symmetric, symmetric, negative_symmetric, half_band, hilbert};
static const char* dds_coeff_struct_str[] = {
    "inferred", "non_symmetric", "symmetric",
    "negative_symmetric", "half_band", "hilbert"
};

enum output_rounding_mode {full_precision = 0, truncate_lsbs, non_symmetric_rounding_down,
                           non_symmetric_rounding_up, symmetric_rounding_to_zero,
                           symmetric_rounding_to_infinity, convergent_rounding_to_even,
                           convergent_rounding_to_odd};
static const char* dds_output_rounding_mode_str[] = {
    "full_precision", "truncate_lsbs", "non_symmetric_rounding_down",
    "non_symmetric_rounding_up", "symmetric_rounding_to_zero",
    "symmetric_rounding_to_infinity", "convergent_rounding_to_even",
    "convergent_rounding_to_odd"
};

enum filter_arch {systolic_multiply_accumulate = 0, transpose_multiply_accumulate};
static const char* dds_filter_arch_str[] = {
    "systolic_multiply_accumulate", "transpose_multiply_accumulate"
};

enum optimization_goal {area = 0, speed};
static const char* dds_opt_goal_str[] = {"area", "speed"};

enum config_sync_mode {on_vector = 0, on_packet};
static const char* dds_s_config_sync_mode_str[] = {"on_vector", "on_packet"};

enum config_method {single = 0, by_channel};
static const char* dds_s_config_method_str[] = {"single", "by_channel"};

struct params_t {
  static const unsigned Phase_Width = 16;
  static const unsigned Output_Width = 16;
  static const unsigned phase_fractional_bits = 0;
  static const unsigned output_fractional_bits = 0;
  static const unsigned num_channels = 1;
  static const unsigned input_length = 128;

  /**
   * dds_compiler_v6_0 Core Generics
   *
   * These are the only generics that influence the operation of this bit-accurate model.
   */
  static const unsigned int PartsPresent = XIP_DDS_PHASE_GEN_AND_SIN_COS_LUT;
#ifndef __GXX_EXPERIMENTAL_CXX0X__
  static const double       DDS_Clock_Rate = 20.0;
#else
  static constexpr double   DDS_Clock_Rate = 20.0;
#endif
  static const unsigned int Channels = 1;
  static const unsigned int Mode_of_Operation = XIP_DDS_MOO_CONVENTIONAL;
  static const unsigned int Modulus = 200;
  static const unsigned int ParameterEntry = XIP_DDS_HARDWARE_PARAMS;
#ifndef __GXX_EXPERIMENTAL_CXX0X__
  static const double       Spurious_Free_Dynamic_Range = 20.0;
  static const double       Frequency_Resolution = 10.0;
#else
  static constexpr double   Spurious_Free_Dynamic_Range = 20.0;
  static constexpr double   Frequency_Resolution = 10.0;
#endif
  static const unsigned int Noise_Shaping = XIP_DDS_NS_NONE;
  static const unsigned int Phase_Increment = XIP_DDS_PINCPOFF_FIXED;
  static const unsigned int Phase_Offset = XIP_DDS_PINCPOFF_NONE;
  static const unsigned int Resync = XIP_DDS_ABSENT;
  static const unsigned int Output_Selection = XIP_DDS_OUT_SIN_AND_COS;
  static const unsigned int Negative_Sine = XIP_DDS_ABSENT;
  static const unsigned int Negative_Cosine = XIP_DDS_ABSENT;
  static const unsigned int Amplitude_Mode = XIP_DDS_FULL_RANGE;
  static const unsigned int Memory_Type = XIP_DDS_MEM_AUTO;
  static const unsigned int Optimization_Goal = XIP_DDS_OPTGOAL_AUTO;
  static const unsigned int DSP48_Use = XIP_DDS_DSP_MIN;
  static const unsigned int Has_TREADY = XIP_DDS_ABSENT;
  static const unsigned int S_CONFIG_Sync_Mode = XIP_DDS_CONFIG_SYNC_VECTOR;
  static const unsigned int Output_Form = XIP_DDS_OUTPUT_TWOS;
  static const unsigned int Latency_Configuration = XIP_DDS_LATENCY_AUTO;
  static const unsigned int Latency = 5;
  static const xip_dds_v6_0_data PINC[XIP_DDS_CHANNELS_MAX];
  static const xip_dds_v6_0_data POFF[XIP_DDS_CHANNELS_MAX];

  /* The following parameters are the results of resolution fns. They are included here so that they are in
     the return structure of get_config */

  static const double       resSFDR;
  static const double       resFrequency_Resolution;
  static const unsigned int resNoise_Shaping;
  static const unsigned int resMemory_Type;
  static const unsigned int resOptimization_Goal;
  static const signed   int resLatency;
  static const unsigned int resPhase_Width;
  static const unsigned int resOutput_Width;
  static const unsigned int resPhaseAngleWidth;
  static const unsigned int resPhaseErrWidth;
  //static const xip_uint resLatencyPipe[ci_max_pipe_stages+1]; //100 is much bigger than max possible latency
};

#ifndef AESL_SYN
//---------------------------------------------------------------------------------------------------------------------
// Example message handler
static void msg_print(void* handle, int error, const char* msg)
{
    printf("%s\n",msg);
}
#endif
} // namespace hls::ip_dds


template<typename CONFIG_T>
class DDS {
public:
    static const unsigned input_axi_width = (CONFIG_T::Phase_Width%8) ? (CONFIG_T::Phase_Width/8+1)*8 : CONFIG_T::Phase_Width;    
    static const unsigned output_axi_width = (CONFIG_T::Output_Width%8) ? (CONFIG_T::Output_Width/8+1)*8 : CONFIG_T::Output_Width;    

    typedef ap_uint<CONFIG_T::Phase_Width> config_t;
    typedef ap_ufixed<input_axi_width, input_axi_width - CONFIG_T::phase_fractional_bits> out_phase_t;

    typedef ap_fixed<output_axi_width, output_axi_width - CONFIG_T::output_fractional_bits>  out_data_t;
private:

#ifndef AESL_SYN
    //// Define array helper functions for types used
    //DEFINE_XIP_ARRAY(real);
    //DEFINE_XIP_ARRAY(complex);
    //DEFINE_XIP_ARRAY(uint);
    //DEFINE_XIP_ARRAY(mpz);
    //DEFINE_XIP_ARRAY(mpz_complex);

    //DEFINE_DDS_XIP_ARRAY(real);
    //DEFINE_DDS_XIP_ARRAY(mpz);
    //DEFINE_DDS_XIP_ARRAY(mpz_complex);

    xip_dds_v6_0* mDDS;
#endif

#ifndef AESL_SYN
    void printConfig(const xip_dds_v6_0_config* cfg)
    {
        printf("Configuration of %s:\n",cfg->name);
        printf("\tDDS       : ");
#if 0
        if (cfg->filter_type == hls::ip_dds::single_rate || 
            cfg->filter_type == hls::ip_dds::hilbert ) {
          printf("%s\n",hls::ip_dds::dds_filter_type_str[cfg->filter_type]);
        } else if ( cfg->filter_type == hls::ip_dds::interpolation ) {
          printf("%s by %d\n",hls::ip_dds::dds_filter_type_str[cfg->filter_type],cfg->zero_pack_factor);
        } else {
          printf("%s up by %d down by %d\n",hls::ip_dds::dds_filter_type_str[cfg->filter_type],cfg->interp_rate,cfg->decim_rate);
        }
        printf("\tCoefficients : %d ",cfg->coeff_sets);
        if ( cfg->is_halfband ) {
          printf("Halfband ");
        }
        if (cfg->reloadable) {
          printf("Reloadable ");
        }
        printf("coefficient set(s) of %d taps\n",cfg->num_coeffs);
        printf("\tData         : %d path(s) of %d %s channel(s)\n",cfg->num_paths,cfg->num_channels,hls::ip_dds::dds_channel_sequence_str[cfg->chan_seq]);
#endif
    }

    void checkModulus () {
        if (CONFIG_T::Mode_of_Operation == XIP_DDS_MOO_RASTERIZED && (CONFIG_T::Modulus < 129 || CONFIG_T::Modulus > 225)) {
            std::cerr << ip_dds::ddsErrChkHead << "Validation failed for parameter Modulus for DDS Compiler. Value \""
                      << CONFIG_T::Modulus << "\" is out of the range [129, 256] supported by Vivado HLS."
                      << std::endl;
            exit(1);
        }
    }

    void checkParamEntry () {
        if (CONFIG_T::ParameterEntry == XIP_DDS_SYSTEM_PARAMS) {
            std::cerr << ip_dds::ddsErrChkHead << "PrameterEntry is not supported for setting XIP_DDS_SYSTEM_PARAMS in Vivado HLS."
                      << std::endl;
            exit(1);
        }
    }

    void checkPartsPresent () {
        if (CONFIG_T::PartsPresent != XIP_DDS_PHASE_GEN_AND_SIN_COS_LUT) {
            std::cerr << ip_dds::ddsErrChkHead << "PartsPresent is only supported for setting XIP_DDS_PHASE_GEN_AND_SIN_COS_LUT in Vivado HLS."
                      << std::endl;
            exit(1);
        }
    }

    void gen_ip_inst()
    {

        xip_dds_v6_0_config dds_cnfg;
        xip_status status = xip_dds_v6_0_default_config(&dds_cnfg);
        dds_cnfg.name =  "dds_compiler";

        dds_cnfg.PartsPresent = CONFIG_T::PartsPresent;
        dds_cnfg.DDS_Clock_Rate = CONFIG_T::DDS_Clock_Rate;
        dds_cnfg.Channels = CONFIG_T::Channels;
        dds_cnfg.Mode_of_Operation = CONFIG_T::Mode_of_Operation;
        dds_cnfg.Modulus = CONFIG_T::Modulus;
        dds_cnfg.ParameterEntry = CONFIG_T::ParameterEntry;
        dds_cnfg.Spurious_Free_Dynamic_Range =  CONFIG_T::Spurious_Free_Dynamic_Range;
        dds_cnfg.Frequency_Resolution = CONFIG_T::Frequency_Resolution;
        dds_cnfg.Noise_Shaping = CONFIG_T::Noise_Shaping;
        dds_cnfg.Phase_Increment = CONFIG_T::Phase_Increment;
        dds_cnfg.Resync = CONFIG_T::Resync;
        dds_cnfg.Phase_Offset = CONFIG_T::Phase_Offset;
        dds_cnfg.Output_Selection = CONFIG_T::Output_Selection;
        dds_cnfg.Negative_Sine = CONFIG_T::Negative_Sine;
        dds_cnfg.Negative_Cosine = CONFIG_T::Negative_Cosine;
        dds_cnfg.Amplitude_Mode = CONFIG_T::Amplitude_Mode;
        dds_cnfg.Memory_Type = CONFIG_T::Memory_Type;
        dds_cnfg.Optimization_Goal = CONFIG_T::Optimization_Goal;
        dds_cnfg.DSP48_Use = CONFIG_T::DSP48_Use;
        dds_cnfg.Has_TREADY = CONFIG_T::Has_TREADY;
        dds_cnfg.S_CONFIG_Sync_Mode = CONFIG_T::S_CONFIG_Sync_Mode;
        dds_cnfg.Output_Form = CONFIG_T::Output_Form;
        dds_cnfg.Latency_Configuration = CONFIG_T::Latency_Configuration;
        for(int mi=0; mi<XIP_DDS_CHANNELS_MAX; mi++) {
          dds_cnfg.PINC[mi] = CONFIG_T::PINC[mi];
          dds_cnfg.POFF[mi] = CONFIG_T::POFF[mi];
        }
        dds_cnfg.Latency = CONFIG_T::Latency;
        dds_cnfg.Phase_Width = CONFIG_T::Phase_Width;
        dds_cnfg.Output_Width = CONFIG_T::Output_Width;

        //Create filter instances
        mDDS = xip_dds_v6_0_create(&dds_cnfg, &ip_dds::msg_print, 0);
        if (!mDDS) {
            printf("ERROR: Cannot create an instance of %s due to an incompatible setting.\n",dds_cnfg.name);
            exit(1);
        } 

        #ifdef DEBUG
        printConfig(&dds_cnfg);
        #endif
    }
#endif

    void insert_spec() {
#ifdef AESL_SYN
        #pragma HLS inline self
            _ssdm_op_SpecKeepValue(
                //"component_name", "dds_compiler_0",
                "gui_behaviour", "Coregen",
                "Amplitude_Mode", CONFIG_T::Amplitude_Mode,
                "Channels", CONFIG_T::Channels,
                "DATA_Has_TLAST", "Not_Required",
                "DDS_Clock_Rate", CONFIG_T::DDS_Clock_Rate,
                "DSP48_Use", CONFIG_T::DSP48_Use,
                "Frequency_Resolution", CONFIG_T::Frequency_Resolution,
                "Has_ACLKEN", "true",
                "Has_ARESETn", "true",
                "Has_Phase_Out", 1,
                "Has_TREADY", "true",
                "Latency", CONFIG_T::Latency,
                "Latency_Configuration", CONFIG_T::Latency_Configuration,
                "Memory_Type", CONFIG_T::Memory_Type,
                "Mode_of_Operation", CONFIG_T::Mode_of_Operation,
                "Modulus", CONFIG_T::Modulus,
                "Negative_Cosine", CONFIG_T::Negative_Cosine,
                "Negative_Sine", CONFIG_T::Negative_Sine,
                "Noise_Shaping", CONFIG_T::Noise_Shaping,
                //"OUTPUT_FORM",
                "Optimization_Goal", CONFIG_T::Optimization_Goal,
                "Output_Selection", CONFIG_T::Output_Selection,
                "Output_Width", CONFIG_T::Output_Width,
                "Phase_Width", CONFIG_T::Phase_Width,
                "Parameter_Entry", CONFIG_T::ParameterEntry,
		        "Spurious_Free_Dynamic_Range", CONFIG_T::Spurious_Free_Dynamic_Range,
                "PartsPresent", CONFIG_T::PartsPresent,
                "Phase_Increment", CONFIG_T::Phase_Increment,
                "Phase_offset", CONFIG_T::Phase_Offset,
                "Resync", CONFIG_T::Resync,
                "PINC1",  CONFIG_T::PINC[0],
                "PINC2",  CONFIG_T::PINC[1],
                "PINC3",  CONFIG_T::PINC[2],
                "PINC4",  CONFIG_T::PINC[3],
                "PINC5",  CONFIG_T::PINC[4],
                "PINC6",  CONFIG_T::PINC[5],
                "PINC7",  CONFIG_T::PINC[6],
                "PINC8",  CONFIG_T::PINC[7],
                "PINC9",  CONFIG_T::PINC[8],
                "PINC10", CONFIG_T::PINC[9],
                "PINC11", CONFIG_T::PINC[10],
                "PINC12", CONFIG_T::PINC[11],
                "PINC13", CONFIG_T::PINC[12],
                "PINC14", CONFIG_T::PINC[13],
                "PINC15", CONFIG_T::PINC[14],
                "PINC16", CONFIG_T::PINC[15],
                "POFF1",  CONFIG_T::POFF[0],
                "POFF2",  CONFIG_T::POFF[1],
                "POFF3",  CONFIG_T::POFF[2],
                "POFF4",  CONFIG_T::POFF[3],
                "POFF5",  CONFIG_T::POFF[4],
                "POFF6",  CONFIG_T::POFF[5],
                "POFF7",  CONFIG_T::POFF[6],
                "POFF8",  CONFIG_T::POFF[7],
                "POFF9",  CONFIG_T::POFF[8],
                "POFF10", CONFIG_T::POFF[9],
                "POFF11", CONFIG_T::POFF[10],
                "POFF12", CONFIG_T::POFF[11],
                "POFF13", CONFIG_T::POFF[12],
                "POFF14", CONFIG_T::POFF[13],
                "POFF15", CONFIG_T::POFF[14],
                "POFF16", CONFIG_T::POFF[15],
                "S_PHASE_Has_TUSER", "Not_Required"
            );
#endif
    }


#ifndef AESL_SYN
    enum sim_mode_t {dataonly, configonly, reloadable};    

    void run_sim (
            ip_dds::in_config_pinc_poff<CONFIG_T> config[CONFIG_T::input_length*CONFIG_T::num_channels],
            ip_dds::out_data_sin_cos<CONFIG_T>  data[CONFIG_T::input_length*CONFIG_T::num_channels],
            out_phase_t phase[CONFIG_T::input_length*CONFIG_T::num_channels]
        )
    {
        //////////////////////////////////////////////
        // C level simulation models for hls::dds
        //////////////////////////////////////////////
  const int number_of_samples = CONFIG_T::input_length;

  xip_dds_v6_0_config config_ret;
  xip_dds_v6_0_config_pkt* pinc_poff_config;

  // Can we read back the updated configuration correctly?
  if (xip_dds_v6_0_get_config(mDDS, &config_ret) != XIP_STATUS_OK) {
    std::cerr << "ERROR: Could not retrieve C model configuration" << std::endl;
  }

  // Set up some arrays to hold values for programmable phase increment and phase offset
  xip_dds_v6_0_data pinc[XIP_DDS_CHANNELS_MAX];
  xip_dds_v6_0_data poff[XIP_DDS_CHANNELS_MAX];

  //------------------------------------------------------------------------------------
  // Set up fields and reserve memory for data and config packets, for this configuration
  //------------------------------------------------------------------------------------

  //Calculate the number of input fields
  xip_uint no_of_input_fields = 0;
  if (config_ret.PartsPresent == XIP_DDS_SIN_COS_LUT_ONLY) {
    no_of_input_fields++; //Phase_In
    assert(0 && "DDS IP doesn't support SIN_COS_LUT_ONLY mode");
  } else {
    if (config_ret.Phase_Increment == XIP_DDS_PINCPOFF_STREAM) {
      no_of_input_fields++; //PINC
      if (config_ret.Resync == XIP_DDS_PRESENT) {
        no_of_input_fields++; //RESYNC
      }
    }
    if (config_ret.Phase_Offset == XIP_DDS_PINCPOFF_STREAM) {
      no_of_input_fields++; //POFF
    }
  }

  //Calculate the number of output fields
  xip_uint no_of_output_fields = 0; //phase output is not optional in the c model.
  if (config_ret.PartsPresent != XIP_DDS_SIN_COS_LUT_ONLY) {
    no_of_output_fields = 1; //PHASE_OUT
  }
  if (config_ret.PartsPresent != XIP_DDS_PHASE_GEN_ONLY) {
    if (config_ret.Output_Selection == XIP_DDS_OUT_SIN_ONLY) no_of_output_fields++; //SIN
    if (config_ret.Output_Selection == XIP_DDS_OUT_COS_ONLY) no_of_output_fields++; //COS
    if (config_ret.Output_Selection == XIP_DDS_OUT_SIN_AND_COS) no_of_output_fields += 2; //SIN and COS
  }

  // Create and allocate memory for I/O structures
  // Create request and response structures

  // Create config packet - pass pointer by reference
  if (config_ret.PartsPresent != XIP_DDS_SIN_COS_LUT_ONLY && (config_ret.Phase_Increment == XIP_DDS_PINCPOFF_PROG || config_ret.Phase_Offset == XIP_DDS_PINCPOFF_PROG)) {
    if (xip_dds_v6_0_alloc_config_pkt(&pinc_poff_config, config_ret.Channels, config_ret.Channels) == XIP_STATUS_OK) {
      std::cout << "INFO: Reserved memory for config packet" << std::endl;
    } else {
      std::cerr << "ERROR: Unable to reserve memory for config packet" << std::endl;
      exit(1);
    }
  }

  // Create input data packet
  xip_array_real* din = xip_array_real_create();
  xip_array_real_reserve_dim(din,3); //dimensions are (Number of samples, channels, PINC/POFF/Phase)
  din->dim_size = 3;
  din->dim[0] = number_of_samples;    //FIXME: number of sample
  din->dim[1] = config_ret.Channels;
  din->dim[2] = no_of_input_fields;
  din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
  if (xip_array_real_reserve_data(din,din->data_size) == XIP_STATUS_OK) {
    std::cout << "INFO: Reserved memory for request as [" << number_of_samples << "," << config_ret.Channels << "," << no_of_input_fields << "] array " << std::endl;
  } else {
    std::cout << "ERROR: Unable to reserve memory for input data packet!" << std::endl;
    exit(2);
  }

  // Request memory for output data
  xip_array_real* dout = xip_array_real_create();
  xip_array_real_reserve_dim(dout,3); //dimensions are (Number of samples, channels, PINC/POFF/Phase)
  dout->dim_size = 3;
  dout->dim[0] = number_of_samples;
  dout->dim[1] = config_ret.Channels;
  dout->dim[2] = no_of_output_fields;
  dout->data_size = dout->dim[0] * dout->dim[1] * dout->dim[2];
  if (xip_array_real_reserve_data(dout,dout->data_size) == XIP_STATUS_OK) {
    std::cout << "INFO: Reserved memory for response as [" << number_of_samples << "," << config_ret.Channels << "," << no_of_output_fields << "] array " << std::endl;
  } else {
    std::cout << "ERROR: Unable to reserve memory for output data packet!" << std::endl;
    exit(3);
  }


  //---------------------------------------------------
  // Populate the input structure with example data
  //---------------------------------------------------
  size_t sample = 0;
  size_t field = 0;
  xip_dds_v6_0_data value = 0.0;

  // Set up pinc and poff, and call config routine, if either phase increment or phase offset is programmable
  if (config_ret.PartsPresent != XIP_DDS_SIN_COS_LUT_ONLY) {
    if (config_ret.Phase_Increment == XIP_DDS_PINCPOFF_PROG || config_ret.Phase_Offset == XIP_DDS_PINCPOFF_PROG) {

      for (int channel = 0; channel < config_ret.Channels; channel++) {
        if (config_ret.Phase_Increment == XIP_DDS_PINCPOFF_PROG) {
          //field is PINC
          //if (config_ret.Mode_of_Operation == XIP_DDS_MOO_RASTERIZED) {
          //  pinc[channel] = rand() % (config_ret.Modulus); // Allow zero too
          //} else {
          //  pinc[channel] = rand() % (1ULL << (config_ret.resPhase_Width)); // Allow zero too
          //}
          pinc[channel] = config[channel].get_pinc();  //FIXME ?: CONFIG_T::PINC[channel];
        }
        if (config_ret.Phase_Offset == XIP_DDS_PINCPOFF_PROG) {
          //field is POFF
          //if (config_ret.Mode_of_Operation == XIP_DDS_MOO_RASTERIZED) {
          //  poff[channel] = (channel + 1) % (config_ret.Modulus);
          //} else {
          //  poff[channel] = (channel + 1) % (1ULL << (config_ret.resPhase_Width)); // Allow zero too
          //}
          poff[channel] = config[channel].get_poff();  //FIXME ?: CONFIG_T::POFF[channel];
        }
      }

      // Copy our local pinc/poff data into the memory we allocated in the config structure
      // If not present, leave the initial values
      if (config_ret.Phase_Increment == XIP_DDS_PINCPOFF_PROG) {
        memcpy(pinc_poff_config->din_pinc, pinc, config_ret.Channels*sizeof(xip_dds_v6_0_data));
      }
      if (config_ret.Phase_Offset == XIP_DDS_PINCPOFF_PROG) {
        memcpy(pinc_poff_config->din_poff, poff, config_ret.Channels*sizeof(xip_dds_v6_0_data));
      }

      // Run the config routine
      if (xip_dds_v6_0_config_do(mDDS, pinc_poff_config) == XIP_STATUS_OK) {
        std::cout << "INFO: config_do was successful" << std::endl;
      } else {
        std::cerr << "ERROR: config_packet failed" << std::endl;
        exit(4);
      }
    }
  }

  //------------------------------------------------------
  //transforming data
  //int resync_sample = rand() % (number_of_samples-2) + 1; // Do a resync randomly in the frame between 2nd and 2nd-last sample
  for(sample = 0; sample < number_of_samples; sample++) {
    for (int channel = 0; channel < config_ret.Channels; channel++) {

      field = 0;

      // Phase_In input, for Sin/Cos LUT configuration only
      if (config_ret.PartsPresent == XIP_DDS_SIN_COS_LUT_ONLY) {
        //field is PHASE_IN
        assert(0 && "DDS IP doesn't support SIN_COS_LUT_ONLY mode");
        //if (config_ret.Mode_of_Operation == XIP_DDS_MOO_RASTERIZED) {
        //  value = rand() % (config_ret.Modulus); // Allow zero too
        //} else {
        //  value = rand() % (1ULL << (config_ret.resPhase_Width)); // Allow zero too
        //}
        //xip_dds_v6_0_xip_array_real_set_data(din, value, sample, channel, field);
        //field++;
      }

      // Streaming phase increment
      if (config_ret.PartsPresent != XIP_DDS_SIN_COS_LUT_ONLY) {
        if (config_ret.Phase_Increment == XIP_DDS_PINCPOFF_STREAM) {
          ////field is PINC
          //if (config_ret.Mode_of_Operation == XIP_DDS_MOO_RASTERIZED) {
          //  value = rand() % (config_ret.Modulus); // Allow zero too
          //} else {
          //  value = rand() % (1ULL << (config_ret.resPhase_Width)); // Allow zero too
          //}
          value = config[sample*config_ret.Channels+channel].get_pinc();
          xip_dds_v6_0_xip_array_real_set_data(din, value, sample, channel, field);
          field++;
        }
      }

      // Streaming phase offset
      if (config_ret.PartsPresent != XIP_DDS_SIN_COS_LUT_ONLY) {
        if (config_ret.Phase_Offset == XIP_DDS_PINCPOFF_STREAM) {
          ////field is POFF
          //if (config_ret.Mode_of_Operation == XIP_DDS_MOO_RASTERIZED) {
          //  value = (channel + 1 + sample) % (config_ret.Modulus);
          //} else {
          //  value = (channel + 1 + sample) % (1ULL << (config_ret.resPhase_Width));
          //}
          value = config[sample*config_ret.Channels+channel].get_poff();
          xip_dds_v6_0_xip_array_real_set_data(din, value, sample, channel, field);
          field++;
        }
      }

#if 0
      // Finally do resync, if required
      if (config_ret.PartsPresent != XIP_DDS_SIN_COS_LUT_ONLY) {
        if ((config_ret.Phase_Increment == XIP_DDS_PINCPOFF_STREAM) && (config_ret.Resync == XIP_DDS_PRESENT)){
          //field is Resync
          if (sample == resync_sample) {
            value = 1;
          } else {
            value = 0;
          }
          xip_dds_v6_0_xip_array_real_set_data(din, value, sample, channel, field);
          field++;
        }
      }
#endif

    }
  }

  //------------------
  // Simulate the core
  //------------------
  std::cout << "INFO: Running the C model..." << std::endl;

  if (xip_dds_v6_0_data_do(mDDS,   //pointer to c model instance
                           din, //pointer to input data structure
                           dout, //pointer to output structure
                           number_of_samples, //first dimension of either data structure
                           config_ret.Channels, //2nd dimension of either data structure
                           no_of_input_fields, //3rd dimension of input
                           no_of_output_fields //3rd dimension of output
                           ) != XIP_STATUS_OK)  {
    std::cerr << "ERROR: C model did not complete successfully" << std::endl;
    xip_array_real_destroy(din);
    xip_array_real_destroy(dout);
    xip_dds_v6_0_destroy(mDDS);
    if (config_ret.PartsPresent != XIP_DDS_SIN_COS_LUT_ONLY && (config_ret.Phase_Increment == XIP_DDS_PINCPOFF_PROG || config_ret.Phase_Offset == XIP_DDS_PINCPOFF_PROG)) {
      xip_dds_v6_0_free_config_pkt(&pinc_poff_config);
    }
    exit(5);
  }
  else {
    std::cout << "INFO: C model transaction completed successfully" << std::endl;
  }

  // When enabled, this will print the result data to stdout
  const int SCALE_FACTOR = sizeof(int)*CHAR_BIT - config_ret.Output_Width;
  for(int sample = 0;sample< number_of_samples;sample++) {
    std::cout << std::endl << "Sample " << sample;
    for(int chan = 0; chan < config_ret.Channels; chan++) {
      std::cout << std::endl << "Channel " << chan;
      field = 0;
      xip_dds_v6_0_xip_array_real_get_data(dout, &value, sample, chan, field);
      std::cout << ":  out phase = " << value;
      phase[sample*config_ret.Channels+chan] = value;
      field++;
      ip_dds::out_data_sin_cos<CONFIG_T> tmp;
      if(config_ret.PartsPresent != XIP_DDS_SIN_COS_LUT_ONLY ) {
        if (config_ret.Output_Selection != XIP_DDS_OUT_COS_ONLY) {
          xip_dds_v6_0_xip_array_real_get_data(dout, &value, sample, chan, field);
          std::cout << " out sin = " << (((int)value << SCALE_FACTOR) >> SCALE_FACTOR);
          //std::cout << " out sin = " << (out_data_t)value;
          tmp.get_sin() = value;
          field++;
        }
        if (config_ret.Output_Selection != XIP_DDS_OUT_SIN_ONLY) {
          xip_dds_v6_0_xip_array_real_get_data(dout, &value, sample, chan, field);
          std::cout << " out cos = " << (((int)value << SCALE_FACTOR) >> SCALE_FACTOR);
          //std::cout << " out cos = " << (out_data_t)value;
          tmp.get_cos() = value;
        }
      }
      data[sample*config_ret.Channels+chan] = tmp;
      std::cout << std::endl;
    }
   }

  //-----------------
  // Reset the core
  // This will clear the phase accumulator state, and any resync input,
  // but leave any programmed phase increment/phase offset values
  // unchanged.
  //-----------------
  if (xip_dds_v6_0_reset(mDDS) == XIP_STATUS_OK) {
    std::cout << "INFO: C model reset successfully" << std::endl;
  } else {
    std::cout << "ERROR: C model reset did not complete successfully" << std::endl;
    exit(6);
  }

    }   //end of run_sim
#endif

    public:
        DDS()
#ifndef AESL_SYN
        : mDDS(0)
#endif
        {
#ifndef AESL_SYN
            rule_check();
            ///////////// IP parameters legality checking /////////////
            // Check CONFIG_T::ParameterEntry
            checkParamEntry();

            // Check CONFIG_T::PartsPresent
            checkPartsPresent();

            // Check CONFIG_T::Mode_of_Operation and CONFIG_T::Modulus
            checkModulus();

            gen_ip_inst();
#endif
        }

        ~DDS()
        {
            #ifdef AESL_SYN
            #pragma HLS inline 
            #else
            xip_dds_v6_0_destroy(mDDS);
            #endif
        }

        void run(
            ip_dds::in_config_pinc_poff<CONFIG_T> config[CONFIG_T::input_length*CONFIG_T::num_channels],
            ip_dds::out_data_sin_cos<CONFIG_T> data[CONFIG_T::input_length*CONFIG_T::num_channels],
            out_phase_t phase[CONFIG_T::input_length*CONFIG_T::num_channels]
        )
        {
        #ifdef AESL_SYN
            //////////////////////////////////////////////
            // C level synthesis models for hls::dds
            //////////////////////////////////////////////
            #pragma HLS inline off 
            #pragma HLS resource core="Vivado_DDS" variable=return metadata="parameterizable"
            //#pragma HLS function INSTANTIATE variable=mConfigParams
            #pragma HLS interface ap_fifo port=config
            #pragma HLS interface ap_fifo port=data
            #pragma HLS interface ap_fifo port=phase
            #pragma HLS data_pack variable=config
            #pragma HLS data_pack variable=data

            insert_spec();
            for (int i = 0; i < CONFIG_T::input_length; ++i)
            {
                ip_dds::in_config_pinc_poff<CONFIG_T> tmp = config[i];
                ip_dds::out_data_sin_cos<CONFIG_T> outtmp;
                outtmp.data[0] = tmp.get_pinc();
             if(CONFIG_T::Output_Selection>1)
                outtmp.data[1] = tmp.get_poff();
                data[i] = outtmp;
                phase[i]= tmp.get_poff();
            }

        #else
            //coeff_t reload_coeff[CONFIG_T::num_coeffs];
            //for (unsigned int i = 0; i < CONFIG_T::num_coeffs; i++) 
            //    reload_coeff[i] = 0;
            //config_t config[CONFIG_T::num_channels] = {0};
            run_sim(config, data, phase);
        #endif
        }

     //void run(
     //  ip_dds::in_config_pinc_poff_resync<CONFIG_T> config[CONFIG_T::input_length*CONFIG_T::num_channels],
     //  ip_dds::out_data_sin_cos<CONFIG_T> data[CONFIG_T::input_length*CONFIG_T::num_channels],
     //  out_phase_t phase[CONFIG_T::input_length*CONFIG_T::num_channels]
     //   )

#ifndef AESL_SYN
        void rule_check()
        {
             std::cout<<"Channel number: "<<CONFIG_T::Channels<<std::endl;
             std::cout<<"Phase INC mode (1 prog 2 fixed 3 stream): "<< CONFIG_T::Phase_Increment << std::endl;
             std::cout<<"Phase OFF mode (0 NONE 1 prog 2 fixed 3 stream): "<< CONFIG_T::Phase_Offset << std::endl;
             std::cout<<""<<std::endl;

             if(CONFIG_T::Output_Width%8 != 0) {
                 std::cerr << ip_dds::ddsErrChkHead << "Output_Width must be an integer multiple of 8." << std::endl;
                 exit(1);
                 //assert(0 && "Output sin/cos width must be exact multiple of 8."); 
             }

             if(CONFIG_T::Phase_Width%8 != 0) {
                 std::cerr << ip_dds::ddsErrChkHead << "Phase_Width must be an integer multiple of 8." << std::endl;
                 exit(1);
                 //assert(0 && "Phase in/out width must be exact multiple of 8."); 
             }

             if (CONFIG_T::Phase_Increment != CONFIG_T::Phase_Offset) {
                 if (CONFIG_T::Phase_Increment == XIP_DDS_PINCPOFF_PROG
                     || CONFIG_T::Phase_Increment == XIP_DDS_PINCPOFF_STREAM) {
                     std::cerr << ip_dds::ddsErrChkHead << " Currently Vivado HLS just support PINC and POFF both are Programmable, or both not." << std::endl;
                     exit(1);
                     //assert(0 && "PINC and POFF mode is not the same."); 
                 }
             }

             if(CONFIG_T::Channels > 16) {
                 assert(0 && "Channel number > 16");
             }
        }
#endif

        void run(
            ip_dds::out_data_sin_cos<CONFIG_T> data[CONFIG_T::input_length*CONFIG_T::num_channels],
            out_phase_t phase[CONFIG_T::input_length*CONFIG_T::num_channels]
        )
        {
        #ifdef AESL_SYN
            //////////////////////////////////////////////
            // C level synthesis models for hls::dds
            //////////////////////////////////////////////
            #pragma HLS inline off 
            #pragma HLS resource core="Vivado_DDS" variable=return metadata="parameterizable"
            //#pragma HLS function INSTANTIATE variable=mConfigParams
            #pragma HLS interface ap_fifo port=data
            #pragma HLS interface ap_fifo port=phase
            #pragma HLS data_pack variable=data

            insert_spec();

            static ip_dds::out_data_sin_cos<CONFIG_T> lut[CONFIG_T::input_length*CONFIG_T::num_channels];
            for(int i=0; i<(CONFIG_T::input_length*CONFIG_T::num_channels);i++)
            {
                lut[i].data[0] = i*i + 3;
if(CONFIG_T::Output_Selection>1)
                lut[i].data[1] = i*i - 69;
            }

            for (int i = 0; i < CONFIG_T::input_length*CONFIG_T::num_channels; ++i)
            {
#if 0
                data[i].data[0] = lut[i].data[0];
if(CONFIG_T::Output_Selection>1)
                data[i].data[1] = lut[i].data[1];
#else
                data[i] = lut[i];
#endif
                phase[i]= i;//lut[i].data + lut[i].data;
            }

        #else
            assert((CONFIG_T::Phase_Increment==0 || CONFIG_T::Phase_Increment==2) && "Phase_Increment set error");
            ip_dds::in_config_pinc_poff<CONFIG_T> config[CONFIG_T::input_length*CONFIG_T::num_channels];
            run_sim (config, data, phase);
        #endif
        }


};
} // namespace hls

#endif // __cplusplus
#endif // X_HLS_DDS_H

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
