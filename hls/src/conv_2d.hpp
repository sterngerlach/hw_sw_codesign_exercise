
// conv_2d.hpp

#ifndef TOYNET_CONV_2D_HPP
#define TOYNET_CONV_2D_HPP

#include "data_types.hpp"

template <int InCh, int OutCh, int H, int W, int OH, int OW,
          int K, int P, int S>
void Conv2d(const fixed_t x[InCh][H][W],
            fixed_t y[OutCh][OH][OW],
            const fixed_t weight[OutCh][InCh][K][K])
{
  // Naive implementation of the 2D convolution layer
  // `x` is of size (`InCh`, `H`, `W`)
  // `y` is of size (`OutCh`, `OH`, `OW`)
  // `weight` is of size (`OutCh`, `InCh`, `K`, `K`)

#pragma HLS INLINE off

  static_assert(K % 2 == 1, "`K` must be an odd number");
  static_assert((H + 2 * P - K) / S + 1 == OH,
                "Output height is inconsistent with the parameters");
  static_assert((W + 2 * P - K) / S + 1 == OW,
                "Output width is inconsistent with the parameters");

  for (int oc = 0; oc < OutCh; ++oc) {
#pragma HLS PIPELINE off
    for (int oh = 0; oh < OH; ++oh) {
#pragma HLS PIPELINE off
      for (int ow = 0; ow < OW; ++ow) {
#pragma HLS PIPELINE off
        fixed_t val = 0;

        for (int ic = 0; ic < InCh; ++ic) {
#pragma HLS PIPELINE off
          for (int kh = 0; kh < K; ++kh) {
#pragma HLS PIPELINE off
            for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE off
              int ih = oh * S + kh - P;
              int iw = ow * S + kw - P;

              if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                val += x[ic][ih][iw] * weight[oc][ic][kh][kw];
            }
          }
        }

        y[oc][oh][ow] = val;
      }
    }
  }
}

template <int InCh, int OutCh, int H, int W, int OH, int OW,
          int K, int P, int S, int B>
void Conv2d2(const fixed_t x[InCh][H][W],
             fixed_t y[OutCh][OH][OW],
             const fixed_t weight[OutCh][InCh][K][K])
{
  // Parallel implementation of the 2D convolution layer
  // `x` is of size (`InCh`, `H`, `W`)
  // `y` is of size (`OutCh`, `OH`, `OW`)
  // `weight` is of size (`OutCh`, `InCh`, `K`, `K`)

#pragma HLS INLINE off

  static_assert(InCh % B == 0,
                "`InCh` must be a multiple of `B`");
  static_assert(K % 2 == 1, "`K` must be an odd number");
  static_assert((H + 2 * P - K) / S + 1 == OH,
                "Output height is inconsistent with the parameters");
  static_assert((W + 2 * P - K) / S + 1 == OW,
                "Output width is inconsistent with the parameters");

  for (int oc = 0; oc < OutCh; ++oc) {
    for (int oh = 0; oh < OH; ++oh) {
      for (int ow = 0; ow < OW; ++ow) {
        fixed_t val = 0;
        fixed_t vals[B];
#pragma HLS ARRAY_PARTITION variable=vals dim=1 complete

        for (int ic0 = 0; ic0 < InCh; ic0 += B) {
          for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
              int ih = oh * S + kh - P;
              int iw = ow * S + kw - P;

              for (int ic1 = 0; ic1 < B; ++ic1) {
#pragma HLS UNROLL
                int ic = ic0 + ic1;
                fixed_t v0 = (ic0 == 0 && kh == 0 && kw == 0) ?
                  fixed_t(0) : vals[ic1];
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                  vals[ic1] = v0 + x[ic][ih][iw] * weight[oc][ic][kh][kw];
                else
                  vals[ic1] = v0;
              }
            }
          }
        }

        for (int ic1 = 0; ic1 < B; ++ic1)
#pragma HLS UNROLL
          val += vals[ic1];

        y[oc][oh][ow] = val;
      }
    }
  }
}

template <int InCh, int OutCh, int H, int W, int OH, int OW,
          int K, int P, int S, int B>
void Conv2d3(const fixed_t x[InCh][H][W],
             fixed_t y[OutCh][OH][OW],
             const fixed_t weight[OutCh][InCh][K][K])
{
  // Parallel implementation of the 2D convolution layer
  // `x` is of size (`InCh`, `H`, `W`)
  // `y` is of size (`OutCh`, `OH`, `OW`)
  // `weight` is of size (`OutCh`, `InCh`, `K`, `K`)

#pragma HLS INLINE off

  static_assert(OutCh % B == 0,
                "`OutCh` must be a multiple of `B`");
  static_assert(K % 2 == 1, "`K` must be an odd number");
  static_assert((H + 2 * P - K) / S + 1 == OH,
                "Output height is inconsistent with the parameters");
  static_assert((W + 2 * P - K) / S + 1 == OW,
                "Output width is inconsistent with the parameters");

  for (int oc0 = 0; oc0 < OutCh; oc0 += B) {
#pragma HLS PIPELINE off
    for (int oh = 0; oh < OH; ++oh) {
#pragma HLS PIPELINE off
      for (int ow = 0; ow < OW; ++ow) {
#pragma HLS PIPELINE off
        fixed_t vals[B];
#pragma HLS ARRAY_PARTITION variable=vals dim=1 complete

        for (int ic = 0; ic < InCh; ++ic) {
#pragma HLS PIPELINE off
          for (int kh = 0; kh < K; ++kh) {
#pragma HLS PIPELINE off
            for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE off
              int ih = oh * S + kh - P;
              int iw = ow * S + kw - P;

              for (int oc1 = 0; oc1 < B; ++oc1) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL
                int oc = oc0 + oc1;
                fixed_t v0 = (ic == 0 && kh == 0 && kw == 0) ?
                  fixed_t(0) : vals[oc1];
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                  vals[oc1] = v0 + x[ic][ih][iw] * weight[oc][ic][kh][kw];
                else
                  vals[oc1] = v0;
              }
            }
          }
        }

        for (int oc1 = 0; oc1 < B; ++oc1) {
#pragma HLS PIPELINE off
#pragma HLS UNROLL
          int oc = oc0 + oc1;
          y[oc][oh][ow] = vals[oc1];
        }
      }
    }
  }
}

template <int InCh, int OutCh, int H, int W, int OH, int OW,
          int K, int P, int S, int B>
void Conv2d4(const fixed_t x[InCh][H][W],
             fixed_t y[OutCh][OH][OW],
             const fixed_t weight[OutCh][InCh][K][K])
{
  // Parallel implementation of the 2D convolution layer
  // `x` is of size (`InCh`, `H`, `W`)
  // `y` is of size (`OutCh`, `OH`, `OW`)
  // `weight` is of size (`OutCh`, `InCh`, `K`, `K`)

#pragma HLS INLINE off

  static_assert(OutCh % B == 0,
                "`OutCh` must be a multiple of `B`");
  static_assert(K % 2 == 1, "`K` must be an odd number");
  static_assert((H + 2 * P - K) / S + 1 == OH,
                "Output height is inconsistent with the parameters");
  static_assert((W + 2 * P - K) / S + 1 == OW,
                "Output width is inconsistent with the parameters");

  for (int oc0 = 0; oc0 < OutCh; oc0 += B) {
#pragma HLS PIPELINE off
    for (int oh = 0; oh < OH; ++oh) {
#pragma HLS PIPELINE off
      for (int ow = 0; ow < OW; ++ow) {
#pragma HLS PIPELINE off
        fixed_t vals[B];
#pragma HLS ARRAY_PARTITION variable=vals dim=1 complete

        for (int ic = 0; ic < InCh; ++ic) {
#pragma HLS PIPELINE off
          for (int kh = 0; kh < K; ++kh) {
#pragma HLS PIPELINE off
            for (int kw = 0; kw < K; ++kw) {
#pragma HLS PIPELINE II=1
              int ih = oh * S + kh - P;
              int iw = ow * S + kw - P;

              for (int oc1 = 0; oc1 < B; ++oc1) {
// #pragma HLS PIPELINE II=1
#pragma HLS UNROLL
                int oc = oc0 + oc1;
                fixed_t v0 = (ic == 0 && kh == 0 && kw == 0) ?
                  fixed_t(0) : vals[oc1];
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                  vals[oc1] = v0 + x[ic][ih][iw] * weight[oc][ic][kh][kw];
                else
                  vals[oc1] = v0;
              }
            }
          }
        }

        for (int oc1 = 0; oc1 < B; ++oc1) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL
          int oc = oc0 + oc1;
          y[oc][oh][ow] = vals[oc1];
        }
      }
    }
  }
}

#endif // TOYNET_CONV_2D_HPP
