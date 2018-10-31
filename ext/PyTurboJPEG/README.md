# PyTurboJPEG
A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image.

## Prerequisites
- [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo/releases)
- [numpy](https://github.com/numpy/numpy)

## Example

```python
import cv2
from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY

# specifying library path explicitly
# jpeg = TurboJPEG(r'D:\turbojpeg.dll')
# jpeg = TurboJPEG('/usr/lib64/libturbojpeg.so')
# jpeg = TurboJPEG('/usr/local/lib/libturbojpeg.dylib')

# using default library installation
jpeg = TurboJPEG()

# decoding input.jpg to BGR array
in_file = open('input.jpg', 'rb')
bgr_array = jpeg.decode(in_file.read())
in_file.close()
cv2.imshow('bgr_array', bgr_array)
cv2.waitKey(0)

# direct rescaling 1/2 while decoding input.jpg to BGR array
in_file = open('input.jpg', 'rb')
bgr_array_half = jpeg.decode(in_file.read(), scaling_factor=(1, 2))
in_file.close()
cv2.imshow('bgr_array_half', bgr_array_half)
cv2.waitKey(0)

# encoding BGR array to output.jpg with default settings.
out_file = open('output.jpg', 'wb')
out_file.write(jpeg.encode(bgr_array))
out_file.close()

# encoding BGR array to output.jpg with TJSAMP_GRAY subsample.
out_file = open('output_gray.jpg', 'wb')
out_file.write(jpeg.encode(bgr_array, jpeg_subsample=TJSAMP_GRAY))
out_file.close()

# encoding BGR array to output.jpg with quality level 50. 
out_file = open('output_quality_50.jpg', 'wb')
out_file.write(jpeg.encode(bgr_array, quality=50))
out_file.close()

# decoding input.jpg to grayscale array
in_file = open('input.jpg', 'rb')
gray_array = jpeg.decode(in_file.read(), pixel_format=TJPF_GRAY)
in_file.close()
cv2.imshow('gray_array', gray_array)
cv2.waitKey(0)
```

## Installation

### macOS
- brew install jpeg-turbo
- pip install -U git+git://github.com/lilohuang/PyTurboJPEG.git

### Windows 
- Download [libjpeg-turbo official installer](https://sourceforge.net/projects/libjpeg-turbo/files) 
- pip install -U git+git://github.com/lilohuang/PyTurboJPEG.git

### Linux
- Download [libjpeg-turbo.repo](https://libjpeg-turbo.org/pmwiki/uploads/Downloads/libjpeg-turbo.repo) to /etc/yum.repos.d/
- sudo yum install libjpeg-turbo-official
- pip install -U git+git://github.com/lilohuang/PyTurboJPEG.git


## Benchmark 

### macOS
- macOS Sierra 10.12.6
- Intel(R) Core(TM) i5-3210M CPU @ 2.50GHz
- opencv-python 3.4.0.12 (pre-built)
- turbo-jpeg 1.5.3 (pre-built)

| Function              | Wall-clock time |
| ----------------------|-----------------|
| cv2.imdecode()        |   0.528 sec     |
| TurboJPEG.decode()    |   0.191 sec     |
| cv2.imencode()        |   0.875 sec     |
| TurboJPEG.encode()    |   0.176 sec     |

### Windows 
- Windows 7 Ultimate 64-bit
- Intel(R) Xeon(R) E3-1276 v3 CPU @ 3.60 GHz
- opencv-python 3.4.0.12 (pre-built)
- turbo-jpeg 1.5.3 (pre-built)

| Function              | Wall-clock time |
| ----------------------|-----------------|
| cv2.imdecode()        |   0.358 sec     |
| TurboJPEG.decode()    |   0.135 sec     |
| cv2.imencode()        |   0.581 sec     |
| TurboJPEG.encode()    |   0.140 sec     |
