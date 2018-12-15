# Debugging

## Common Error Messages

| Error Message  | Failing Component | Resolution
| ------------- | ------------- | ------------- | 
| OSError: libxilinxopencl.so: cannot open shared object file | [xdnn.py](../../xfdnn/rt/xdnn.py) | This shared object file is located at runtime by use of the XILINX_OPENCL environment variable. To ensure this variable is set properly do: source ml-suite/overlaybins/setup.sh <platform> where platform is either aws or 1525
|   File "ml-suite/xfdnn/rt/xdnn_io.py", line 123, in loadImageBlobFromFile: height, width, _ = img.shape : AttributeError: 'NoneType' object has no attribute 'shape' | xdnn_io.py | xdnn_io.py does not have any error checking to ensure that a loaded file is actually an image. This was done for speed, but can be a burden if you accidentally pass a text file to the API. Ensure you have only images passed in.|
