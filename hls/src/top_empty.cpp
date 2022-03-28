
// top_empty.cpp

#include "data_types.hpp"

void InferenceEmpty(hls::stream<axi_stream_data_t>& in_stream,
                    hls::stream<axi_stream_data_t>& out_stream)
{
#pragma HLS INTERFACE axis register_mode=both register port=in_stream
#pragma HLS INTERFACE axis register_mode=both register port=out_stream
#pragma HLS INTERFACE ap_ctrl_none port=return

  axi_stream_data_t in_data = in_stream.read();

  axi_stream_data_t out_data;
  out_data.data = in_data.data + 13;
  out_data.keep = in_data.keep;
  out_data.strb = in_data.strb;
  out_data.last = in_data.last;
  out_data.user = in_data.user;
  out_data.id = in_data.id;
  out_data.dest = in_data.dest;

  out_stream.write(out_data);
}
