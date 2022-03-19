
// max_pool_2d.hpp

#ifndef TOYNET_MAX_POOL_2D_HPP
#define TOYNET_MAX_POOL_2D_HPP

#include "data_types.hpp"

template <int C, int H, int W, int K>
void MaxPool2d(const fixed_t x[C][H][W],
               fixed_t y[C][H / K][W / K])
{
  // Naive implementation of the 2D max-pooling layer
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C`, `H/K`, `W/K`)

#pragma HLS INLINE off

  static_assert(H % K == 0, "`H` must be a multiple of `K`");
  static_assert(W % K == 0, "`W` must be a multiple of `K`");

  for (int c = 0; c < C; ++c) {
#pragma HLS PIPELINE off
    for (int oh = 0; oh < H / K; ++oh) {
#pragma HLS PIPELINE off
      for (int ow = 0; ow < W / K; ++ow) {
#pragma HLS PIPELINE off
        fixed_t val;

        for (int kh = 0; kh < K; ++kh) {
#pragma HLS PIPELINE off
          for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE off
            int ih = oh * K + kh;
            int iw = ow * K + kw;

            if (kh == 0 && kw == 0)
              val = x[c][ih][iw];
            else
              val = val > x[c][ih][iw] ? val : x[c][ih][iw];
          }
        }

        y[c][oh][ow] = val;
      }
    }
  }
}

template <int C, int H, int W, int K, int B>
void MaxPool2d2(const fixed_t x[C][H][W],
                fixed_t y[C][H / K][W / K])
{
  // Parallel implementation of the 2D max-pooling layer
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C`, `H/K`, `W/K`)

#pragma HLS INLINE off

  static_assert(H % K == 0, "`H` must be a multiple of `K`");
  static_assert(W % K == 0, "`W` must be a multiple of `K`");

  for (int c0 = 0; c0 < C; c0 += B) {
#pragma HLS PIPELINE off
    for (int oh = 0; oh < H / K; ++oh) {
#pragma HLS PIPELINE off
      for (int ow = 0; ow < W / K; ++ow) {
#pragma HLS PIPELINE off
        fixed_t vals[B];
#pragma HLS ARRAY_PARTITION variable=vals dim=1 complete

        for (int kh = 0; kh < K; ++kh) {
#pragma HLS PIPELINE off
          for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE off
            int ih = oh * K + kh;
            int iw = ow * K + kw;

            for (int c1 = 0; c1 < B; ++c1) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL
              int c = c0 + c1;
              if (kh == 0 && kw == 0)
                vals[c1] = x[c][ih][iw];
              else
                vals[c1] = vals[c1] > x[c][ih][iw] ? vals[c1] : x[c][ih][iw];
            }
          }
        }

        for (int c1 = 0; c1 < B; ++c1) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL
          int c = c0 + c1;
          y[c][oh][ow] = vals[c1];
        }
      }
    }
  }
}

template <int C, int H, int W, int K, int B>
void MaxPool2d3(const fixed_t x[C][H][W],
                fixed_t y[C][H / K][W / K])
{
  // Parallel implementation of the 2D max-pooling layer
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C`, `H/K`, `W/K`)

#pragma HLS INLINE off

  static_assert(H % K == 0, "`H` must be a multiple of `K`");
  static_assert(W % K == 0, "`W` must be a multiple of `K`");

  for (int c0 = 0; c0 < C; c0 += B) {
#pragma HLS PIPELINE off
    for (int oh = 0; oh < H / K; ++oh) {
#pragma HLS PIPELINE off
      for (int ow = 0; ow < W / K; ++ow) {
#pragma HLS PIPELINE off
        fixed_t vals[B];
#pragma HLS ARRAY_PARTITION variable=vals dim=1 complete

        for (int kh = 0; kh < K; ++kh) {
#pragma HLS PIPELINE off
          for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=1
            int ih = oh * K + kh;
            int iw = ow * K + kw;

            for (int c1 = 0; c1 < B; ++c1) {
// #pragma HLS PIPELINE II=1
#pragma HLS UNROLL
              int c = c0 + c1;
              if (kh == 0 && kw == 0)
                vals[c1] = x[c][ih][iw];
              else
                vals[c1] = vals[c1] > x[c][ih][iw] ? vals[c1] : x[c][ih][iw];
            }
          }
        }

        for (int c1 = 0; c1 < B; ++c1) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL
          int c = c0 + c1;
          y[c][oh][ow] = vals[c1];
        }
      }
    }
  }
}

#endif // TOYNET_MAX_POOL_2D_HPP
