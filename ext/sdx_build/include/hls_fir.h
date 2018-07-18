/* -*- c++ -*-*/
/*
 * __VIVADO_HLS_COPYRIGHT-INFO__ 
 *
 *
 */

#ifndef X_HLS_FIR_H
#define X_HLS_FIR_H

/*
 * This file contains a C++ model of hls::fir.
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
#include "fir/fir_compiler_v7_2_bitacc_cmodel.h"
#endif

namespace hls {

#ifdef AESL_SYN
#include "etc/autopilot_ssdm_op.h"
#endif

namespace ip_fir {

#ifndef INLINE
#define INLINE inline __attribute__((always_inline))
#endif

static const char* firErrChkHead = "ERROR:hls::fir ";

enum filter_type {single_rate = 0, interpolation, decimation, hilbert_filter, interpolated};
static const char* fir_filter_type_str[] = {
    "single_rate", "interpolation", 
    "decimation", "hilbert", "interpolated"
};

enum rate_change_type {integer = 0, fixed_fractional};
static const char* fir_rate_change_type_str[] = {
    "integer", "fixed_fractional"
};

enum chan_seq {basic = 0, advanced};
static const char* fir_channel_sequence_str[] = {
    "basic", "advanced"
};

enum rate_specification {frequency = 0, input_period, output_period};
static const char* fir_ratespecification_str[] = {
    "frequency_specification", "input_sample_period", "output_sample_period"
};

enum value_sign {value_signed = 0, value_unsigned};
static const char* fir_value_sign_str[] = {"signed", "unsigned"};

enum quantization {integer_coefficients = 0, quantize_only, maximize_dynamic_range};
static const char* fir_quantization_str[] = {
    "integer_coefficients", "quantize_only", "maximize_dynamic_range"
};

enum coeff_structure {inferred = 0, non_symmetric, symmetric, negative_symmetric, half_band, hilbert};
static const char* fir_coeff_struct_str[] = {
    "inferred", "non_symmetric", "symmetric",
    "negative_symmetric", "half_band", "hilbert"
};

enum output_rounding_mode {full_precision = 0, truncate_lsbs, non_symmetric_rounding_down,
                           non_symmetric_rounding_up, symmetric_rounding_to_zero,
                           symmetric_rounding_to_infinity, convergent_rounding_to_even,
                           convergent_rounding_to_odd};
static const char* fir_output_rounding_mode_str[] = {
    "full_precision", "truncate_lsbs", "non_symmetric_rounding_down",
    "non_symmetric_rounding_up", "symmetric_rounding_to_zero",
    "symmetric_rounding_to_infinity", "convergent_rounding_to_even",
    "convergent_rounding_to_odd"
};

enum filter_arch {systolic_multiply_accumulate = 0, transpose_multiply_accumulate};
static const char* fir_filter_arch_str[] = {
    "systolic_multiply_accumulate", "transpose_multiply_accumulate"
};

enum optimization_goal {area = 0, speed};
static const char* fir_opt_goal_str[] = {"area", "speed"};

enum config_sync_mode {on_vector = 0, on_packet};
static const char* fir_s_config_sync_mode_str[] = {"on_vector", "on_packet"};

enum config_method {single = 0, by_channel};
static const char* fir_s_config_method_str[] = {"single", "by_channel"};

struct params_t {
    static const unsigned input_width = 16;
    static const unsigned input_fractional_bits = 0;
    static const unsigned output_width = 24;
    static const unsigned output_fractional_bits = 0;
    static const unsigned coeff_width = 16;
    static const unsigned coeff_fractional_bits = 0;
    static const unsigned config_width = 8;
    static const unsigned num_coeffs = 21;
    static const unsigned coeff_sets = 1;
    static const unsigned input_length = num_coeffs;
    static const unsigned output_length = num_coeffs;
    static const unsigned num_channels = 1;

    static const unsigned total_num_coeff = num_coeffs * coeff_sets;
    static const double coeff_vec[total_num_coeff];
    static const bool reloadable = false;
    static const unsigned filter_type = single_rate;
    static const unsigned rate_change = integer;
    static const unsigned interp_rate = 1;
    static const unsigned decim_rate = 1;
    static const unsigned zero_pack_factor = 1;
    static const unsigned chan_seq = basic;
    static const unsigned rate_specification = input_period;
    static const unsigned sample_period = 1;
#ifndef __GXX_EXPERIMENTAL_CXX0X__
    static const double sample_frequency = 0.001;
#else
    static constexpr double sample_frequency = 0.001;
#endif
    static const unsigned quantization = integer_coefficients;
    static const bool best_precision = false;
    static const unsigned coeff_structure = non_symmetric;
    static const unsigned output_rounding_mode = full_precision;
    static const unsigned filter_arch = systolic_multiply_accumulate;
    static const unsigned optimization_goal = area;
    static const unsigned inter_column_pipe_length = 4;
    static const unsigned column_config = 1;
    static const unsigned config_sync_mode = on_vector;
    static const unsigned config_method = single;
    static const unsigned coeff_padding = 0;

    static const unsigned num_paths = 1;
    static const unsigned data_sign = value_signed;
    static const unsigned coeff_sign = value_signed;
};

#ifndef AESL_SYN
//---------------------------------------------------------------------------------------------------------------------
// Example message handler
static void msg_print(void* handle, int error, const char* msg)
{
    printf("%s\n",msg);
}
#endif
} // namespace hls::ip_fir

using namespace std;

template<typename CONFIG_T>
class FIR {
private:
    static const unsigned input_axi_width = ((CONFIG_T::input_width+7)>>3)<<3;    
    static const unsigned output_axi_width = ((CONFIG_T::output_width+7)>>3)<<3;    
    static const unsigned coeff_axi_width = ((CONFIG_T::coeff_width+7)>>3)<<3;    

    typedef ap_fixed<input_axi_width, input_axi_width - CONFIG_T::input_fractional_bits> in_data_t;
    typedef ap_fixed<output_axi_width, output_axi_width - CONFIG_T::output_fractional_bits>  out_data_t;
    typedef ap_uint<CONFIG_T::config_width> config_t;
    typedef ap_fixed<coeff_axi_width, coeff_axi_width - CONFIG_T::coeff_fractional_bits> coeff_t;

#ifndef AESL_SYN
    //// Define array helper functions for types used
    //DEFINE_XIP_ARRAY(real);
    //DEFINE_XIP_ARRAY(complex);
    //DEFINE_XIP_ARRAY(uint);
    //DEFINE_XIP_ARRAY(mpz);
    //DEFINE_XIP_ARRAY(mpz_complex);

    //DEFINE_FIR_XIP_ARRAY(real);
    //DEFINE_FIR_XIP_ARRAY(mpz);
    //DEFINE_FIR_XIP_ARRAY(mpz_complex);

    xip_fir_v7_2* mFIR;
#endif

#ifndef AESL_SYN
    void printConfig(const xip_fir_v7_2_config* cfg)
    {
        printf("Configuration of %s:\n",cfg->name);
        printf("\tFilter       : ");
        if (cfg->filter_type == hls::ip_fir::single_rate || 
            cfg->filter_type == hls::ip_fir::hilbert_filter ) {
          printf("%s\n",hls::ip_fir::fir_filter_type_str[cfg->filter_type]);
        } else if ( cfg->filter_type == hls::ip_fir::interpolation ) {
          printf("%s by %d\n",hls::ip_fir::fir_filter_type_str[cfg->filter_type],cfg->zero_pack_factor);
        } else {
          printf("%s up by %d down by %d\n",hls::ip_fir::fir_filter_type_str[cfg->filter_type],cfg->interp_rate,cfg->decim_rate);
        }
        printf("\tCoefficients : %d ",cfg->coeff_sets);
        if ( cfg->is_halfband ) {
          printf("Halfband ");
        }
        if (cfg->reloadable) {
          printf("Reloadable ");
        }
        printf("coefficient set(s) of %d taps\n",cfg->num_coeffs);
        printf("\tData         : %d path(s) of %d %s channel(s)\n",cfg->num_paths,cfg->num_channels,hls::ip_fir::fir_channel_sequence_str[cfg->chan_seq]);
    }

    void gen_ip_inst()
    {

        xip_fir_v7_2_config fir_cnfg;
        fir_cnfg.name =  "fir_compiler";

        fir_cnfg.coeff = &CONFIG_T::coeff_vec[0];
        fir_cnfg.filter_type = CONFIG_T::filter_type;
        fir_cnfg.rate_change = CONFIG_T::rate_change;
        fir_cnfg.interp_rate = CONFIG_T::interp_rate;
        fir_cnfg.decim_rate = CONFIG_T::decim_rate;
        fir_cnfg.zero_pack_factor = CONFIG_T::zero_pack_factor;
        fir_cnfg.num_channels = CONFIG_T::num_channels;
        fir_cnfg.coeff_sets = CONFIG_T::coeff_sets;
        fir_cnfg.num_coeffs = CONFIG_T::num_coeffs;
        fir_cnfg.reloadable = CONFIG_T::reloadable;
        fir_cnfg.quantization = CONFIG_T::quantization;
        fir_cnfg.coeff_width = CONFIG_T::coeff_width;
        fir_cnfg.coeff_fract_width = CONFIG_T::coeff_fractional_bits;
        fir_cnfg.chan_seq = CONFIG_T::chan_seq;
        fir_cnfg.data_width = CONFIG_T::input_width;
        fir_cnfg.data_fract_width = CONFIG_T::input_fractional_bits;
        fir_cnfg.output_rounding_mode = CONFIG_T::output_rounding_mode;
        fir_cnfg.output_width = CONFIG_T::output_width; 
        fir_cnfg.output_fract_width = CONFIG_T::output_fractional_bits;
        fir_cnfg.config_method = CONFIG_T::config_method;
        fir_cnfg.coeff_padding = CONFIG_T::coeff_padding;
        fir_cnfg.is_halfband = (CONFIG_T::coeff_structure == ip_fir::half_band);

        //FIXME: doesn't support the following params
        fir_cnfg.init_pattern = P4_3;
        fir_cnfg.num_paths = 1; 

        //Create filter instances
        mFIR = xip_fir_v7_2_create(&fir_cnfg, &ip_fir::msg_print, 0);
        if (!mFIR) {
            printf("Error creating instance %s\n",fir_cnfg.name);
            return;
        } 

        #ifdef DEBUG
        printConfig(&fir_cnfg);
        #endif
    }
#endif

    void insert_spec() {
#ifdef AESL_SYN
        #pragma HLS inline self
            _ssdm_op_SpecKeepValue(
                //"component_name", "fir_compiler_0",
                "gui_behaviour", "Coregen",
                "coefficientsource", "Vector",
                "coefficientvector", CONFIG_T::coeff_vec,
                "coefficient_file", "no_coe_file_loaded",
                "coefficient_sets", CONFIG_T::coeff_sets,
                "coefficient_reload", CONFIG_T::reloadable,
                "filter_type", CONFIG_T::filter_type,
                "rate_change_type", CONFIG_T::rate_change,
                "interpolation_rate", CONFIG_T::interp_rate,
                "decimation_rate", CONFIG_T::decim_rate,
                "zero_pack_factor", CONFIG_T::zero_pack_factor,
                "channel_sequence", CONFIG_T::chan_seq,
                "number_channels", CONFIG_T::num_channels,
                "select_pattern", "All",
                "pattern_list", "P4-0,P4-1,P4-2,P4-3,P4-4",
                "number_paths", CONFIG_T::num_paths,
                "ratespecification", CONFIG_T::rate_specification,
                "sampleperiod", CONFIG_T::sample_period,
                "sample_frequency", CONFIG_T::sample_frequency,
                "clock_frequency", "300.0",
                "coefficient_sign", CONFIG_T::coeff_sign,
                "quantization", CONFIG_T::quantization,
                "coefficient_width", CONFIG_T::coeff_width,
                "bestprecision", CONFIG_T::best_precision,
                "coefficient_fractional_bits", CONFIG_T::coeff_fractional_bits,
                "coefficient_structure", CONFIG_T::coeff_structure,
                "data_sign", CONFIG_T::data_sign,
                "data_width", CONFIG_T::input_width,
                "data_fractional_bits", CONFIG_T::input_fractional_bits,
                "output_rounding_mode", CONFIG_T::output_rounding_mode,
                "output_width", CONFIG_T::output_width,
                "filter_architecture", CONFIG_T::filter_arch,
                "optimization_goal", CONFIG_T::optimization_goal,
                "optimization_selection", "None",
                "optimization_list", "None",
                "data_buffer_type", "Automatic",
                "coefficient_buffer_type", "Automatic",
                "input_buffer_type", "Automatic",
                "output_buffer_type", "Automatic",
                "preference_for_other_storage", "Automatic",
                "multi_column_support", "Automatic",
                "inter_column_pipe_length", CONFIG_T::inter_column_pipe_length,
                "columnconfig", CONFIG_T::column_config,
                "data_has_tlast", "Packet_Framing",
                "m_data_has_tready", "true",
                "s_data_has_fifo", "true",
                "s_data_has_tuser", "Not_Required",
                "m_data_has_tuser", "Not_Required",
                "data_tuser_width", "1",
                "s_config_sync_mode", CONFIG_T::config_sync_mode,
                "s_config_method", CONFIG_T::config_method,
                "num_reload_slots", "1",
                "has_aclken", "true",
                "has_aresetn", "true",
                "reset_data_vector", "true",
                "gen_mif_from_spec", "false",
                "gen_mif_from_coe", "false",
                "reload_file", "no_coe_file_loaded",
                "gen_mif_files", "false",
                "displayreloadorder", "false",
                "passband_min", "0.0",
                "passband_max", "0.5",
                "stopband_min", "0.5",
                "stopband_max", "1.0",
                "filter_selection", "1"
            );
#endif
    }


#ifndef AESL_SYN
    enum sim_mode_t {dataonly, configonly, reloadable};    

    void run_sim (
        in_data_t in[CONFIG_T::input_length * CONFIG_T::num_channels],
        out_data_t out[CONFIG_T::output_length * CONFIG_T::num_channels],
        config_t config[CONFIG_T::num_channels],
        coeff_t reload[CONFIG_T::num_coeffs + ((CONFIG_T::coeff_sets == 1) ? 0 : 1)],
        sim_mode_t mode)
    {
        //////////////////////////////////////////////
        // C level simulation models for hls::fir
        //////////////////////////////////////////////
        // Create input data packet
        xip_array_real* din = xip_array_real_create();
        xip_array_real_reserve_dim(din,3);
        din->dim_size = 3; // 3D array
        din->dim[0] = 1;
        din->dim[1] = CONFIG_T::num_channels;
        din->dim[2] = CONFIG_T::input_length;
        din->data_size = din->dim[0] * din->dim[1] * din->dim[2];
        if (xip_array_real_reserve_data(din,din->data_size) != XIP_STATUS_OK) {
            printf("Unable to reserve data!\n");
            return;
        }
        
        // Create output data packet
        //  - Automatically sized using xip_fir_v7_2_calc_size
        xip_array_real* fir_out = xip_array_real_create();
        xip_array_real_reserve_dim(fir_out,3);
        fir_out->dim_size = 3; // 3D array

        if(xip_fir_v7_2_calc_size(mFIR,din,fir_out,0)== XIP_STATUS_OK) {
            if (xip_array_real_reserve_data(fir_out,fir_out->data_size) != XIP_STATUS_OK) {
                printf("Unable to reserve data!\n");
                return;
            }
        } else {
            printf("Unable to calculate output date size\n");
            return;
        }

        //FIXME: check and promote msg
        assert(fir_out->data_size == CONFIG_T::output_length * CONFIG_T::num_channels);

        // Populate data in with an impulse
        // FIXME: assume path=1 and chan = 1
        for (unsigned idx = 0; idx < CONFIG_T::input_length; ++idx)
        {
            for (unsigned chan = 0; chan < CONFIG_T::num_channels; ++chan)
                xip_fir_v7_2_xip_array_real_set_chan(din, in[idx * CONFIG_T::num_channels + chan], 0, chan, idx, P_BASIC);
        }

        //#define DEBUG
        #ifdef DEBUG
        std::cout << "s_sata" << std::endl;
        for (int i=0; i< din->data_size; i++)
            std::cout << " " << din->data[i] ;
        std::cout << std::endl;
        #endif

        // send new configuration
        xip_array_uint* fsel = 0; 
        if ((mode == configonly) || (mode == reloadable))
        {
            assert(CONFIG_T::coeff_sets > 1 || CONFIG_T::reloadable);
            // Create config packet
            xip_array_uint* fsel = xip_array_uint_create();
            xip_array_uint_reserve_dim(fsel,1);
            fsel->dim_size = 1;
            fsel->dim[0] = CONFIG_T::num_channels;
            fsel->data_size = fsel->dim[0];
            if (xip_array_uint_reserve_data(fsel,fsel->data_size) != XIP_STATUS_OK) {
                printf("Unable to reserve data!\n");
                return;
            }

            xip_fir_v7_2_cnfg_packet cnfg;
            cnfg.fsel = fsel;
            for (unsigned i = 0; i < cnfg.fsel->data_size; ++i)
                cnfg.fsel->data[i] = config[i].to_int();

            // Send config data
            if (xip_fir_v7_2_config_send(mFIR, &cnfg) != XIP_STATUS_OK) {
                printf("Error sending config packet\n");
                return;
            }

            #ifdef DEBUG
            std::cout << "Config packet: " ;
            for (int i = 0; i < cnfg.fsel->data_size; ++i)
                std::cout << " " << cnfg.fsel->data[i];
            std::cout << std::endl; 
            #endif
        }

        xip_fir_v7_2_rld_packet rld;
        // send reloaded coefficients
        if (mode == reloadable)
        {
            assert(CONFIG_T::reloadable);
            if (CONFIG_T::coeff_sets == 1)
                rld.fsel = 0;
            else
                rld.fsel = reload[0];
            rld.coeff = xip_array_real_create();
            xip_array_real_reserve_dim(rld.coeff,1);
            rld.coeff->dim_size=1;
            rld.coeff->dim[0]=CONFIG_T::num_coeffs;
            rld.coeff->data_size = rld.coeff->dim[0];
            if (xip_array_real_reserve_data(rld.coeff,rld.coeff->data_size) != XIP_STATUS_OK) {
                printf("Unable to reserve coeff!\n");
                return;
            }

            // Copy coefficients into reload packet
            int coeff_i;
            bool isAllZero = true;
            int coeff_offset = (CONFIG_T::coeff_sets == 1) ? 0 : 1;
            for (coeff_i= 0; coeff_i < CONFIG_T::num_coeffs; coeff_i++) { 
                rld.coeff->data[coeff_i] = (xip_real)(reload[coeff_i + coeff_offset]); 
                isAllZero &= (reload[coeff_i] == 0);
            }

            // Send reload data
            if (!isAllZero) {
                if ( xip_fir_v7_2_reload_send(mFIR, &rld) != XIP_STATUS_OK) {
                    printf("Error sending reload packet\n");
                    return;
                }

                #ifdef DEBUG
                std::cout << "Reload packet: ";
                if (CONFIG_T::coeff_sets > 1)
                    std::cout << "fsel = " << rld.fsel << "\t; new coeff : ";
                for (int i = 0; i < rld.coeff->data_size; ++i)
                    std::cout << " " << rld.coeff->data[i];
                std::cout << std::endl;
                #endif
            }
        }           
        
        // Send input data and filter
        if ( xip_fir_v7_2_data_send(mFIR,din)!= XIP_STATUS_OK) {
            printf("Error sending data\n");
            return;
        } 

        // Retrieve filtered data
        if (xip_fir_v7_2_data_get(mFIR,fir_out,0) != XIP_STATUS_OK) {
            printf("Error getting data\n");
            return;
        }

        // FIXME: assume path=1 and chan = 1
        for (unsigned idx = 0; idx < CONFIG_T::output_length; ++idx)
        {
            for (unsigned chan = 0; chan < CONFIG_T::num_channels; ++chan)
            {
                xip_real val;
                xip_fir_v7_2_xip_array_real_get_chan(fir_out, &val, 0, chan, idx, P_BASIC);
                out[idx * CONFIG_T::num_channels+ chan] = (out_data_t)val; 
            }
        }

        //DEBUG
        #ifdef DEBUG
        std::cout << "m_sata" << std::endl;
        for (int i=0; i< fir_out->data_size; i++)
            std::cout << " " << fir_out->data[i] ;
        std::cout << std::endl;
        #endif

        xip_array_real_destroy(din);
        xip_array_real_destroy(fir_out);
        if (fsel) xip_array_uint_destroy(fsel);
        if (mode == reloadable) xip_array_real_destroy(rld.coeff);
    }
#endif

    public:
        FIR()
#ifndef AESL_SYN
        : mFIR(0)
#endif
        {
#ifndef AESL_SYN
            gen_ip_inst();
#endif
        }

        ~FIR()
        {
            #ifdef AESL_SYN
            #pragma HLS inline 
            #else
            xip_fir_v7_2_destroy(mFIR);
            #endif
        }

        void run(
            in_data_t in[CONFIG_T::input_length * CONFIG_T::num_channels],
            out_data_t out[CONFIG_T::output_length * CONFIG_T::num_channels]
        )
        {
        #ifdef AESL_SYN
            //////////////////////////////////////////////
            // C level synthesis models for hls::fir
            //////////////////////////////////////////////
            #pragma HLS inline off 
            #pragma HLS resource core="Vivado_FIR" variable=return metadata="parameterizable"
            //#pragma HLS function INSTANTIATE variable=mConfigParams
            #pragma HLS interface ap_fifo port=in
            #pragma HLS interface ap_fifo port=out
            #pragma HLS data_pack variable=in
            #pragma HLS data_pack variable=out

            insert_spec();
            for (int i = 0; i < CONFIG_T::input_length; ++i)
                out[i] = in[i];

        #else
            coeff_t reload_coeff[CONFIG_T::num_coeffs];
            for (unsigned int i = 0; i < CONFIG_T::num_coeffs; i++) 
                reload_coeff[i] = 0;
            config_t config[CONFIG_T::num_channels] = {0};
            run_sim(in, out, config, reload_coeff, dataonly);
        #endif
        }

        void run(
            in_data_t in[CONFIG_T::input_length * CONFIG_T::num_channels],
            out_data_t out[CONFIG_T::output_length * CONFIG_T::num_channels],
            config_t config[CONFIG_T::num_channels])
        {
        #ifdef AESL_SYN
            //////////////////////////////////////////////
            // C level synthesis models for hls::fir
            //////////////////////////////////////////////
            #pragma HLS inline off 
            #pragma HLS resource core="Vivado_FIR" variable=return metadata="parameterizable"
            //#pragma HLS function INSTANTIATE variable=mConfigParams
            #pragma HLS interface ap_fifo port=in
            #pragma HLS interface ap_fifo port=out
            #pragma HLS interface ap_fifo port=config
            #pragma HLS data_pack variable=in
            #pragma HLS data_pack variable=out

            insert_spec();
            if (*config)
                for (int i = 0; i < CONFIG_T::input_length; ++i)
                    out[i] = in[i];

        #else
            coeff_t reload_coeff[CONFIG_T::num_coeffs];
            for (unsigned int i = 0; i < CONFIG_T::num_coeffs; i++) 
                reload_coeff[i] = 0;
            run_sim(in, out, config, reload_coeff, configonly);
        #endif
        }


        void run(
            in_data_t in[CONFIG_T::input_length * CONFIG_T::num_channels],
            out_data_t out[CONFIG_T::output_length * CONFIG_T::num_channels],
            config_t config[CONFIG_T::num_channels],
            coeff_t reload[CONFIG_T::num_coeffs + ((CONFIG_T::coeff_sets == 1) ? 0 : 1)])
        {
        #ifdef AESL_SYN
            //////////////////////////////////////////////
            // C level synthesis models for hls::fir
            //////////////////////////////////////////////
            #pragma HLS inline off 
            #pragma HLS resource core="Vivado_FIR" variable=return metadata="parameterizable"
            //#pragma HLS function INSTANTIATE variable=mConfigParams
            #pragma HLS interface ap_fifo port=in
            #pragma HLS interface ap_fifo port=out
            #pragma HLS interface ap_fifo port=config
            #pragma HLS interface ap_fifo port=reload
            #pragma HLS data_pack variable=in
            #pragma HLS data_pack variable=out

            insert_spec();
            if (*config)
                for (int i = 0; i < CONFIG_T::input_length; ++i)
                    out[i] = in[i] + reload[i & CONFIG_T::num_coeffs];
        #else
            run_sim(in, out, config, reload, reloadable);
        #endif
        }


};
} // namespace hls

#endif // __cplusplus
#endif // X_HLS_FIR_H

// 67d7842dbbe25473c3c32b93c0da8047785f30d78e8a024de1b57352245f9689
