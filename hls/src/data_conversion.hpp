
// data_conversion.hpp

#ifndef TOYNET_DATA_CONVERSION_HPP
#define TOYNET_DATA_CONVERSION_HPP

#include <cstdint>

union float_u32_t
{
  float f;
  std::uint32_t u;
};

// Interpret float as std::uint32_t
inline std::uint32_t FloatToU32(float f)
{
  union float_u32_t f_u32;
  f_u32.f = f;
  return f_u32.u;
}

// Interpret std::uint32_t as float
inline float U32ToFloat(std::uint32_t u)
{
  union float_u32_t f_u32;
  f_u32.u = u;
  return f_u32.f;
}

#endif // TOYNET_DATA_CONVERSION_HPP
