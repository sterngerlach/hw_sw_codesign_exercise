
// data_bit_params.hpp

#ifndef TOYNET_DATA_BIT_PARAMS_HPP
#define TOYNET_DATA_BIT_PARAMS_HPP

#ifndef BIT_WIDTH
#warning Bit width is not defined (default: 32)
// Data width of the model parameters
constexpr int kBitWidth = 32;
#else
// Data width of the model parameters
constexpr int kBitWidth = BIT_WIDTH;
#endif // BIT_WIDTH

#ifndef INT_BIT_WIDTH
#warning Bit width (integer part) is not defined (default: 16)
// Number of integer bits
constexpr int kIntegerBitWidth = 16;
#else
// Number of integer bits
constexpr int kIntegerBitWidth = INT_BIT_WIDTH;
#endif // INT_BIT_WIDTH

#endif // TOYNET_DATA_BIT_PARAMS_HPP
