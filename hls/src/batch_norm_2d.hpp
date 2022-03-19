
// batch_norm_2d.hpp

#ifndef TOYNET_BATCH_NORM_2D_HPP
#define TOYNET_BATCH_NORM_2D_HPP

#include "data_types.hpp"

template <int C, int H, int W>
void BatchNorm2dReLU(const fixed_t x[C][H][W],
                     fixed_t y[C][H][W],
                     const fixed_t scale[C],
                     const fixed_t bias[C],
                     const fixed_t mean[C])
{
  // Naive implementation of the batch normalization and ReLU activation
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C`, `H`, `W`)
  // `scale` is of size (`C`)
  // `bias` is of size (`C`)
  // `mean` is of size (`C`)

#pragma HLS INLINE off

  for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE off
    for (int h = 0; h < H; ++h) {
#pragma HLS PIPELINE off
      for (int w = 0; w < W; ++w) {
#pragma HLS PIPELINE off
        // Batch normalization with the learned parameters
        fixed_t val = (x[c][h][w] - mean[c]) * scale[c] + bias[c];
        // ReLU activation
        y[c][h][w] = val > fixed_t(0) ? val : fixed_t(0);
      }
    }
  }
}

template <int C, int H, int W, int B>
void BatchNorm2dReLU2(const fixed_t x[C][H][W],
                      fixed_t y[C][H][W],
                      const fixed_t scale[C],
                      const fixed_t bias[C],
                      const fixed_t mean[C])
{
  // Parallel implementation of the batch normalization and ReLU activation
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C`, `H`, `W`)
  // `scale` is of size (`C`)
  // `bias` is of size (`C`)
  // `mean` is of size (`C`)

#pragma HLS INLINE off

  for (int c0 = 0; c0 < C; c0 += B) {
#pragma HLS PIPELINE off
    for (int h = 0; h < H; ++h) {
#pragma HLS PIPELINE off
      for (int w = 0; w < W; ++w) {
#pragma HLS PIPELINE off
        for (int c1 = 0; c1 < B; ++c1) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL
          int c = c0 + c1;
          // Batch normalization with the learned parameters
          fixed_t val = (x[c][h][w] - mean[c]) * scale[c] + bias[c];
          // ReLU activation
          y[c][h][w] = val > fixed_t(0) ? val : fixed_t(0);
        }
      }
    }
  }
}

template <int C, int H, int W, int B>
void BatchNorm2dReLU3(const fixed_t x[C][H][W],
                      fixed_t y[C][H][W],
                      const fixed_t scale[C],
                      const fixed_t bias[C],
                      const fixed_t mean[C])
{
  // Parallel implementation of the batch normalization and ReLU activation
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C`, `H`, `W`)
  // `scale` is of size (`C`)
  // `bias` is of size (`C`)
  // `mean` is of size (`C`)

#pragma HLS INLINE off

  for (int c0 = 0; c0 < C; c0 += B) {
#pragma HLS PIPELINE off
    for (int h = 0; h < H; ++h) {
#pragma HLS PIPELINE off
      for (int w = 0; w < W; ++w) {
#pragma HLS PIPELINE II=1
        for (int c1 = 0; c1 < B; ++c1) {
// #pragma HLS PIPELINE II=1
#pragma HLS UNROLL
          int c = c0 + c1;
          // Batch normalization with the learned parameters
          fixed_t val = (x[c][h][w] - mean[c]) * scale[c] + bias[c];
          // ReLU activation
          y[c][h][w] = val > fixed_t(0) ? val : fixed_t(0);
        }
      }
    }
  }
}

#endif // TOYNET_BATCH_NORM_2D_HPP
