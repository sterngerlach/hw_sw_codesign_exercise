
// top_opt2.cpp

#include "batch_norm_2d.hpp"
#include "conv_2d.hpp"
#include "data_transfer.hpp"
#include "data_types.hpp"
#include "depthwise_conv_2d.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "max_pool_2d.hpp"

void InferenceOpt2Core(hls::stream<axi_stream_data_t>& in_stream,
                       hls::stream<axi_stream_data_t>& out_stream,
                       const fixed_t conv0_weight[6][1][5][5],
                       const fixed_t bn0_scale[6],
                       const fixed_t bn0_bias[6],
                       const fixed_t bn0_mean[6],
                       const fixed_t conv1_weight[16][6][5][5],
                       const fixed_t bn1_scale[16],
                       const fixed_t bn1_bias[16],
                       const fixed_t bn1_mean[16],
                       const fixed_t fc0_weight[120][400],
                       const fixed_t fc0_bias[120],
                       const fixed_t fc1_weight[84][120],
                       const fixed_t fc1_bias[84],
                       const fixed_t fc2_weight[10][84],
                       const fixed_t fc2_bias[10])
{
#pragma HLS INLINE off

  // Input, output, and intermediate results
  fixed_t x0[1][28][28];
  fixed_t x1[6][28][28];
  fixed_t x2[6][14][14];
  fixed_t x3[6][14][14];
  fixed_t x4[16][10][10];
  fixed_t x5[16][5][5];
  fixed_t x6[16][5][5];
  fixed_t x7[400];
  fixed_t x8[120];
  fixed_t x9[84];
  fixed_t x10[10];

#pragma HLS ARRAY_PARTITION variable=x1 dim=1 factor=3 cyclic
#pragma HLS ARRAY_PARTITION variable=x2 dim=1 factor=3 cyclic
#pragma HLS ARRAY_PARTITION variable=x3 dim=1 factor=3 cyclic
#pragma HLS ARRAY_PARTITION variable=x4 dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=x5 dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=x6 dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=x7 dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=x8 dim=1 factor=4 cyclic
#pragma HLS ARRAY_PARTITION variable=x9 dim=1 factor=2 cyclic

  // Read the input
  ReadArray3d<1, 28, 28>(x0, in_stream);

  // Inference
  Conv2d4<1, 6, 28, 28, 28, 28, 5, 2, 1, 6>(x0, x1, conv0_weight);
  MaxPool2d3<6, 28, 28, 2, 6>(x1, x2);
  BatchNorm2dReLU3<6, 14, 14, 6>(x2, x3, bn0_scale, bn0_bias, bn0_mean);
  Conv2d4<6, 16, 14, 14, 10, 10, 5, 0, 1, 16>(x3, x4, conv1_weight);
  MaxPool2d3<16, 10, 10, 2, 16>(x4, x5);
  BatchNorm2dReLU3<16, 5, 5, 16>(x5, x6, bn1_scale, bn1_bias, bn1_mean);
  Flatten3d<16, 5, 5>(x6, x7);
  Linear3<400, 120, true, 16>(x7, fc0_weight, fc0_bias, x8);
  Linear3<120, 84, true, 8>(x8, fc1_weight, fc1_bias, x9);
  Linear3<84, 10, false, 4>(x9, fc2_weight, fc2_bias, x10);

  // Write the output
  WriteArray1d<10>(x10, out_stream);
}

void InferenceOpt2(hls::stream<axi_stream_data_t>& in_stream,
                   hls::stream<axi_stream_data_t>& out_stream)
{
#pragma HLS INTERFACE axis register_mode=both register port=in_stream
#pragma HLS INTERFACE axis register_mode=both register port=out_stream
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Optimized implementation with the loop unrolling and pipelining

  // Model parameters
  fixed_t conv0_weight[6][1][5][5];
  fixed_t bn0_scale[6], bn0_bias[6], bn0_mean[6];
  fixed_t conv1_weight[16][6][5][5];
  fixed_t bn1_scale[16], bn1_bias[16], bn1_mean[16];
  fixed_t fc0_weight[120][400], fc0_bias[120];
  fixed_t fc1_weight[84][120], fc1_bias[84];
  fixed_t fc2_weight[10][84], fc2_bias[10];

#pragma HLS ARRAY_PARTITION variable=conv0_weight dim=1 factor=3 cyclic
#pragma HLS ARRAY_PARTITION variable=bn0_scale dim=1 factor=3 cyclic
#pragma HLS ARRAY_PARTITION variable=bn0_bias dim=1 factor=3 cyclic
#pragma HLS ARRAY_PARTITION variable=bn0_mean dim=1 factor=3 cyclic
#pragma HLS ARRAY_PARTITION variable=conv1_weight dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=bn1_scale dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=bn1_bias dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=bn1_mean dim=1 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=fc0_weight dim=2 factor=8 cyclic
#pragma HLS ARRAY_PARTITION variable=fc1_weight dim=2 factor=4 cyclic
#pragma HLS ARRAY_PARTITION variable=fc2_weight dim=2 factor=2 cyclic

  axi_stream_data_t in_data;
  in_data = in_stream.read();
  const int mode = static_cast<int>(in_data.data.to_int());

  if (mode == kModeInitWeights) {
    // Read the model parameters
    ReadConv2dParams<1, 6, 5>(conv0_weight, in_stream);
    ReadBatchNorm2dParams<6>(bn0_scale, bn0_bias, bn0_mean, in_stream);
    ReadConv2dParams<6, 16, 5>(conv1_weight, in_stream);
    ReadBatchNorm2dParams<16>(bn1_scale, bn1_bias, bn1_mean, in_stream);
    ReadLinearParams<400, 120>(fc0_weight, fc0_bias, in_stream);
    ReadLinearParams<120, 84>(fc1_weight, fc1_bias, in_stream);
    ReadLinearParams<84, 10>(fc2_weight, fc2_bias, in_stream);

    // Write the acknowledgment message
    WriteAck(out_stream);
  } else if (mode == kModeInference) {
    InferenceOpt2Core(in_stream, out_stream,
      conv0_weight, bn0_scale, bn0_bias, bn0_mean,
      conv1_weight, bn1_scale, bn1_bias, bn1_mean,
      fc0_weight, fc0_bias, fc1_weight, fc1_bias,
      fc2_weight, fc2_bias);
  }
}
