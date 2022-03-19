
// top_naive.cpp

#include "batch_norm_2d.hpp"
#include "conv_2d.hpp"
#include "data_transfer.hpp"
#include "data_types.hpp"
#include "depthwise_conv_2d.hpp"
#include "flatten.hpp"
#include "linear.hpp"
#include "max_pool_2d.hpp"

void InferenceNaiveCore(hls::stream<axi_stream_data_t>& in_stream,
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

  // Read the input
  ReadArray3d<1, 28, 28>(x0, in_stream);

  // Inference
  Conv2d<1, 6, 28, 28, 28, 28, 5, 2, 1>(x0, x1, conv0_weight);
  MaxPool2d<6, 28, 28, 2>(x1, x2);
  BatchNorm2dReLU<6, 14, 14>(x2, x3, bn0_scale, bn0_bias, bn0_mean);
  Conv2d<6, 16, 14, 14, 10, 10, 5, 0, 1>(x3, x4, conv1_weight);
  MaxPool2d<16, 10, 10, 2>(x4, x5);
  BatchNorm2dReLU<16, 5, 5>(x5, x6, bn1_scale, bn1_bias, bn1_mean);
  Flatten3d<16, 5, 5>(x6, x7);
  Linear<400, 120, true>(x7, fc0_weight, fc0_bias, x8);
  Linear<120, 84, true>(x8, fc1_weight, fc1_bias, x9);
  Linear<84, 10, false>(x9, fc2_weight, fc2_bias, x10);

  // Write the output
  WriteArray1d<10>(x10, out_stream);
}

void InferenceNaive(hls::stream<axi_stream_data_t>& in_stream,
                    hls::stream<axi_stream_data_t>& out_stream)
{
#pragma HLS INTERFACE axis register_mode=both register port=in_stream
#pragma HLS INTERFACE axis register_mode=both register port=out_stream
#pragma HLS INTERFACE s_axilite port=return bundle=control

  // Model parameters
  fixed_t conv0_weight[6][1][5][5];
  fixed_t bn0_scale[6], bn0_bias[6], bn0_mean[6];
  fixed_t conv1_weight[16][6][5][5];
  fixed_t bn1_scale[16], bn1_bias[16], bn1_mean[16];
  fixed_t fc0_weight[120][400], fc0_bias[120];
  fixed_t fc1_weight[84][120], fc1_bias[84];
  fixed_t fc2_weight[10][84], fc2_bias[10];

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
    InferenceNaiveCore(in_stream, out_stream,
      conv0_weight, bn0_scale, bn0_bias, bn0_mean,
      conv1_weight, bn1_scale, bn1_bias, bn1_mean,
      fc0_weight, fc0_bias, fc1_weight, fc1_bias,
      fc2_weight, fc2_bias);
  }
}
