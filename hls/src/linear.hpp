
// linear.hpp

#ifndef TOYNET_LINEAR_HPP
#define TOYNET_LINEAR_HPP

#include "data_types.hpp"

template <int InDims, int OutDims, bool ApplyReLU>
void Linear(const fixed_t x[InDims],
            const fixed_t weight[OutDims][InDims],
            const fixed_t bias[OutDims],
            fixed_t y[OutDims])
{
  // Naive implementation of the fully-connected layer
  // `x` is of size (1, `InDims`)
  // `weight` is of size (`OutDims`, `InDims`)
  // `bias` is of size (`OutDims`)
  // `y` is of size (1, `OutDims`)

#pragma HLS INLINE off

  for (int i = 0; i < OutDims; ++i) {
#pragma HLS PIPELINE off
    fixed_t val = 0;

    for (int j = 0; j < InDims; ++j)
#pragma HLS PIPELINE off
      val += x[j] * weight[i][j];

    val += bias[i];

    if (ApplyReLU)
      y[i] = val > fixed_t(0) ? val : fixed_t(0);
    else
      y[i] = val;
  }
}

template <int InDims, int OutDims, bool ApplyReLU, int B>
void Linear2(const fixed_t x[InDims],
             const fixed_t weight[OutDims][InDims],
             const fixed_t bias[OutDims],
             fixed_t y[OutDims])
{
  // Parallel implementation of the fully-connected layer
  // Innermost loop is parallelized by a factor of `B`
  // `x` is of size (1, `InDims`)
  // `weight` is of size (`OutDims`, `InDims`)
  // `bias` is of size (`OutDims`)
  // `y` is of size (1, `OutDims`)

#pragma HLS INLINE off

  static_assert(InDims % B == 0,
                "`InDims` must be a multiple of `B`");

  for (int i = 0; i < OutDims; ++i) {
#pragma HLS PIPELINE off
    fixed_t val = 0;
    fixed_t vals[B];
#pragma HLS ARRAY_PARTITION variable=vals dim=1 complete

    for (int j0 = 0; j0 < InDims; j0 += B) {
#pragma HLS PIPELINE off
      for (int j1 = 0; j1 < B; ++j1) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL
        int j = j0 + j1;
        if (j0 == 0)
          vals[j1] = x[j] * weight[i][j];
        else
          vals[j1] += x[j] * weight[i][j];
      }
    }

    for (int j1 = 0; j1 < B; ++j1)
#pragma HLS PIPELINE off
#pragma HLS UNROLL
      val += vals[j1];

    val += bias[i];

    if (ApplyReLU)
      y[i] = val > fixed_t(0) ? val : fixed_t(0);
    else
      y[i] = val;
  }
}

template <int InDims, int OutDims, bool ApplyReLU, int B>
void Linear3(const fixed_t x[InDims],
             const fixed_t weight[OutDims][InDims],
             const fixed_t bias[OutDims],
             fixed_t y[OutDims])
{
  // Parallel implementation of the fully-connected layer
  // Innermost loop is parallelized by a factor of `B`
  // `x` is of size (1, `InDims`)
  // `weight` is of size (`OutDims`, `InDims`)
  // `bias` is of size (`OutDims`)
  // `y` is of size (1, `OutDims`)

#pragma HLS INLINE off

  static_assert(InDims % B == 0,
                "`InDims` must be a multiple of `B`");

  for (int i = 0; i < OutDims; ++i) {
#pragma HLS PIPELINE off
    fixed_t val = 0;
    fixed_t vals[B];
#pragma HLS ARRAY_PARTITION variable=vals dim=1 complete

    for (int j0 = 0; j0 < InDims; j0 += B) {
#pragma HLS PIPELINE II=1
      for (int j1 = 0; j1 < B; ++j1) {
// #pragma HLS PIPELINE II=1
#pragma HLS UNROLL
        int j = j0 + j1;
        if (j0 == 0)
          vals[j1] = x[j] * weight[i][j];
        else
          vals[j1] += x[j] * weight[i][j];
      }
    }

    for (int j1 = 0; j1 < B; ++j1)
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL
      val += vals[j1];

    val += bias[i];

    if (ApplyReLU)
      y[i] = val > fixed_t(0) ? val : fixed_t(0);
    else
      y[i] = val;
  }
}

#endif // TOYNET_LINEAR_HPP
