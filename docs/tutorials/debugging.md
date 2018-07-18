# Debugging

## Common Error Messages

| Error Message  | Failing Component | Resolution
| ------------- | ------------- | ------------- | 
| OSError: libxilinxopencl.so: cannot open shared object file | [xdnn.py](../../xfdnn/rt/xdnn.py) | This shared object file is located at runtime by use of the XILINX_OPENCL environment variable. To ensure this variable is set properly do: source ml-suite/overlaybins/setup.sh <platform> where platform is either aws or 1525
| Content Cell  | Content Cell  |
