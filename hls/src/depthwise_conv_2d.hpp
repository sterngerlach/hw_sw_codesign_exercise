
// depthwise_conv_2d.hpp

#ifndef TOYNET_DEPTHWISE_CONV_2D_HPP
#define TOYNET_DEPTHWISE_CONV_2D_HPP

#include "data_types.hpp"

template <int C, int H, int W, int OH, int OW,
          int K, int P, int S>
void DepthwiseConv2d(const fixed_t x[C][H][W],
                     fixed_t y[C][OH][OW],
                     const fixed_t weight[C][K][K])
{
  // Naive implementation of the depthwise convolution
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C`, `OH`, `OW`)
  // `weight` is of size (`C`, `K`, `K`)

#pragma HLS INLINE off

  static_assert(K % 2 == 1, "`K` must be an odd number");
  static_assert((H + 2 * P - K) / S + 1 == OH,
                "Output height is inconsistent with the parameters");
  static_assert((W + 2 * P - K) / S + 1 == OW,
                "Output width is inconsistent with the parameters");

  for (int c = 0; c < C; ++c) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        fixed_t val = 0;

        for (int kh = 0; kh < K; ++kh) {
          for (int kw = 0; kw < K; ++kw) {
            int ih = oh * S + kh - P;
            int iw = ow * S + kw - P;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
              val += x[c][ih][iw] * weight[c][kh][kw];
          }
        }

        y[c][oh][ow] = val;
      }
    }
  }
}

template <int C, int H, int W, int OH, int OW,
          int K, int P, int S, int B>
void DepthwiseConv2d2(const fixed_t x[C][H][W],
                      fixed_t y[C][OH][OW],
                      const fixed_t weight[C][K][K])
{
  // Parallel implementation of the depthwise convolution
  // `x` is of size (`C`, `H`, `W`)
  // `y` is of size (`C`, `OH`, `OW`)
  // `weight` is of size (`C`, `K`, `K`)

#pragma HLS INLINE off

  static_assert(C % B == 0,
                "`C` must be a multiple of `B`");
  static_assert(K % 2 == 1, "`K` must be an odd number");
  static_assert((H + 2 * P - K) / S + 1 == OH,
                "Output height is inconsistent with the parameters");
  static_assert((W + 2 * P - K) / S + 1 == OW,
                "Output width is inconsistent with the parameters");

  for (int c0 = 0; c0 < C; c0 += B) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        fixed_t vals[B];
#pragma HLS ARRAY_PARTITION variable=vals dim=1 complete

        for (int kh = 0; kh < K; ++kh) {
          for (int kw = 0; kw < K; ++kw) {
            int ih = oh * S + kh - P;
            int iw = ow * S + kw - P;

            for (int c1 = 0; c1 < B; ++c1) {
#pragma HLS UNROLL
              int c = c0 + c1;
              fixed_t v0 = (kh == 0 && kw == 0) ? fixed_t(0) : vals[c1];
              if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                vals[c1] = v0 + x[c][ih][iw] * weight[c][kh][kw];
              else
                vals[c1] = v0;
            }
          }
        }

        for (int c1 = 0; c1 < B; ++c1) {
#pragma HLS UNROLL
          int c = c0 + c1;
          y[c][oh][ow] = vals[c1];
        }
      }
    }
  }
}

#endif // TOYNET_DEPTHWISE_CONV_2D_HPP
