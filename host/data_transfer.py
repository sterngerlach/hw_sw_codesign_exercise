# coding: utf-8
# data_transfer.py

# Example:
# sudo XILINX_XRT=/usr python3 data_transfer.py zcu104_empty.bit

import numpy as np
import sys
import time

from pynq import allocate, Overlay

def main():
    if len(sys.argv) != 2:
        print(f"Usage: ${sys.argv[0]} <Bitstream>")
        sys.exit(1)

    overlay = Overlay(sys.argv[1])

    if not overlay.is_loaded():
        print(f"Failed to load the bitstream: {sys.argv[1]}")
        sys.exit(1)

    dma = overlay.axi_dma

    # 32MiB
    buf_len = 256 * (2 ** 15)
    # 64MiB - 1bit
    # buf_len = 256 * (2 ** 16) - 1
    print(f"Buffer size: {buf_len * 4 / 1024} KiB")
    print(f"Buffer size: {buf_len * 4 / 1024 / 1024} MiB")

    buf_in = allocate(shape=(buf_len,), dtype=np.uint32, cacheable=False)
    buf_out = allocate(shape=(buf_len,), dtype=np.uint32, cacheable=False)

    buf_in[:] = np.random.rand(buf_len) * 1e2

    print("Before: ")
    print(buf_in[:10])
    print(buf_out[:10])
    print(buf_in[-10:])
    print(buf_out[-10:])

    t0 = time.time()

    dma.sendchannel.transfer(buf_in)
    dma.recvchannel.transfer(buf_out)
    dma.recvchannel.wait()

    t1 = time.time()
    elapsed = (t1 - t0) * 1e3
    print("Elapsed time: {:.3f} ms".format(elapsed))

    elapsed = t1 - t0
    throughput = buf_len * 32 / 1024 / 1024 / elapsed * 2
    print(f"Throughput: {throughput} Mbps")

    print("After: ")
    print(buf_in[:10])
    print(buf_out[:10])
    print(buf_in[-10:])
    print(buf_out[-10:])

if __name__ == "__main__":
    main()

