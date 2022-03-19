
// data_types.hpp

#ifndef TOYNET_DATA_TYPES_HPP
#define TOYNET_DATA_TYPES_HPP

#include <ap_fixed.h>
#include <ap_int.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>

#include "data_bit_params.hpp"

// Data width of the AXI4-Stream interface
constexpr int kAxiStreamWidth = 32;
// Data types for AXI4-Stream
using axi_stream_data_t = ap_axiu<kAxiStreamWidth, 0, 0, 0>;

// Value types
using fixed_t = ap_fixed<kBitWidth, kIntegerBitWidth,
                         ap_q_mode::AP_TRN, ap_o_mode::AP_SAT, 0>;

// Operation modes
constexpr int kModeInitWeights = 1;
constexpr int kModeInference = 2;

#endif // TOYNET_DATA_TYPES_HPP
