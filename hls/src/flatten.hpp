
// flatten.hpp

#ifndef TOYNET_FLATTEN_HPP
#define TOYNET_FLATTEN_HPP

#include "data_types.hpp"

template <int C, int H, int W>
void Flatten3d(const fixed_t x[C][H][W],
               fixed_t y[C * H * W])
{
  // Naive implementation of the flatten layer
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C * H * W`)

#pragma HLS INLINE off

  for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE off
    for (int h = 0; h < H; ++h) {
#pragma HLS PIPELINE off
      for (int w = 0; w < W; ++w) {
#pragma HLS PIPELINE off
        y[c * (H * W) + h * W + w] = x[c][h][w];
      }
    }
  }
}

template <int C, int H, int W>
void Flatten3d2(const fixed_t x[C][H][W],
                fixed_t y[C * H * W])
{
  // Naive implementation of the flatten layer
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C * H * W`)

  for (int c = 0; c < C; ++c) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        y[c * (H * W) + h * W + w] = x[c][h][w];
      }
    }
  }
}

#endif // TOYNET_FLATTEN_HPP
