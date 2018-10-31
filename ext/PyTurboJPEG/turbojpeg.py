#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
import os
import os.path as op
import platform
from ctypes import *
from ctypes.util import find_library
from enum import Enum  # N.B. install enum34 if in python < 3.5
import contextlib
import numpy as np


class TJPF(Enum):
    # Pixel formats
    # see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/turbojpeg.h
    UNKNOWN = -1
    RGB = 0
    BGR = 1
    RGBX = 2
    BGRX = 3
    XBGR = 4
    XRGB = 5
    GRAY = 6
    RGBA = 7
    BGRA = 8
    ABGR = 9
    ARGB = 10
    CMYK = 11

    @property
    def pixel_size(self):
        """Pixel size (in bytes) for a given pixel format."""
        return [3, 3, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4, None][self.value]

    @property
    def offsets(self):
        """Offsets (in bytes) for the red, green, blue and alpha components for a given pixel format."""
        r = [0, 2, 0, 2, 3, 1, None, 0, 2, 3, 1, None][self.value]
        g = [1, 1, 1, 1, 2, 2, None, 1, 1, 2, 2, None][self.value]
        b = [2, 0, 2, 0, 1, 3, None, 2, 0, 1, 3, None][self.value]
        a = [None, None, None, None, None, None, None, 3, 3, 0, 0, None][self.value]
        return r, g, b, a


class TJSAMP(Enum):
    # Chrominance subsampling options
    # see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/turbojpeg.h
    YUV444 = 0
    YUV422 = 1
    YUV420 = 2
    GRAY = 3
    YUV440 = 4

    def mcu(self):
        return [(8, 8), (16, 8), (16, 16), (8, 8), (8, 16), (32, 8)][self.value]


class TJFLAG(Enum):
    # Some quality-speed tradeoff and other flags
    # see details in https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/turbojpeg.h
    BOTTOMUP = 2
    FASTUPSAMPLE = 256
    NOREALLOC = 1024
    FASTDCT = 2048
    ACCURATEDCT = 4096
    STOPONWARNING = 8192
    PROGRESSIVE = 16384


class TurboJPEG(object):
    """A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image."""

    def __init__(self, lib_path=None):

        self.lib_path = self._find_turbo(lib_path)
        turbo_jpeg = cdll.LoadLibrary(self.lib_path)

        self.__init_decompress = turbo_jpeg.tjInitDecompress
        self.__init_decompress.restype = c_void_p
        self.__init_compress = turbo_jpeg.tjInitCompress
        self.__init_compress.restype = c_void_p

        self.__decompress_header = turbo_jpeg.tjDecompressHeader3
        self.__decompress_header.argtypes = [
            c_void_p, POINTER(c_ubyte), c_ulong, POINTER(c_int),
            POINTER(c_int), POINTER(c_int), POINTER(c_int)]
        self.__decompress_header.restype = c_int

        self.__decompress = turbo_jpeg.tjDecompress2
        self.__decompress.argtypes = [
            c_void_p, POINTER(c_ubyte), c_ulong, POINTER(c_ubyte),
            c_int, c_int, c_int, c_int, c_int]
        self.__decompress.restype = c_int

        self.__compress = turbo_jpeg.tjCompress2
        self.__compress.argtypes = [
            c_void_p, POINTER(c_ubyte), c_int, c_int, c_int, c_int,
            POINTER(c_void_p), POINTER(c_ulong), c_int, c_int, c_int]
        self.__compress.restype = c_int

        self.__destroy = turbo_jpeg.tjDestroy
        self.__destroy.argtypes = [c_void_p]
        self.__destroy.restype = c_int

        self.__free = turbo_jpeg.tjFree
        self.__free.argtypes = [c_void_p]
        self.__free.restype = None

        self.__get_error_str = turbo_jpeg.tjGetErrorStr
        self.__get_error_str.restype = c_char_p

        self.__scaling_factors = []

        class ScalingFactor(Structure):
            _fields_ = ('num', c_int), ('denom', c_int)
        get_scaling_factors = turbo_jpeg.tjGetScalingFactors
        get_scaling_factors.argtypes = [POINTER(c_int)]
        get_scaling_factors.restype = POINTER(ScalingFactor)
        num_scaling_factors = c_int()
        scaling_factors = get_scaling_factors(byref(num_scaling_factors))
        for i in range(num_scaling_factors.value):
            self.__scaling_factors.append((scaling_factors[i].num, scaling_factors[i].denom))

    # --- Decoding

    @contextlib.contextmanager
    def _decode_handle(self, handle=None):
        # N.B. handlers are infintely reusable
        # But init_compress / init_decompress + destroy costs are usually very small
        # It usually won't be worth the added lifecycle complexity to cache them
        # This little CM is probably slower itself, but makes code a bit nicer to read
        destroy = handle is None
        if handle is None:
            handle = self.__init_decompress()
        yield handle
        if destroy:
            self.__destroy(handle)

    def _info(self, jpeg_buf, handle=None):
        with self._decode_handle(handle) as handle:
            width = c_int()
            height = c_int()
            jpeg_subsample = c_int()
            jpeg_colorspace = c_int()
            jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
            src_addr = jpeg_array.ctypes.data_as(POINTER(c_ubyte))
            status = self.__decompress_header(
                handle, src_addr, jpeg_array.size, byref(width), byref(height),
                byref(jpeg_subsample), byref(jpeg_colorspace))
            if status != 0:
                raise IOError(self.__get_error_str().decode())
            return width, height, jpeg_subsample, jpeg_colorspace, jpeg_array, src_addr

    def info(self, jpeg_buf):
        width, height, jpeg_subsample, jpeg_colorspace, _, _ = self._info(jpeg_buf)
        return width.value, height.value, TJSAMP(jpeg_subsample.value).name, TJPF(jpeg_colorspace.value).name

    def decode(self,
               jpeg_buf,
               pixel_format=TJPF.BGR,
               scaling_factor=None,
               fast_upsample=False,
               fast_dct=False,
               accurate_dct=False,
               bottomup=False):
        """Decode JPEG buffer to numpy array."""
        with self._decode_handle() as handle:
            if scaling_factor is not None and scaling_factor not in self.__scaling_factors:
                raise ValueError('supported scaling factors are ' + str(self.__scaling_factors))

            # Decompress header
            width, height, jpeg_subsample, jpeg_colorspace, jpeg_array, src_addr = self._info(jpeg_buf)

            # Allocate destination memory (we should allow to customize this)
            scaled_width = width.value
            scaled_height = height.value
            if scaling_factor is not None:
                def get_scaled_value(dim, num, denom):
                    return (dim * num + denom - 1) // denom
                scaled_width = get_scaled_value(
                    scaled_width, scaling_factor[0], scaling_factor[1])
                scaled_height = get_scaled_value(
                    scaled_height, scaling_factor[0], scaling_factor[1])
            img_array = np.empty(
                [scaled_height, scaled_width, pixel_format.pixel_size],
                dtype=np.uint8)
            dest_addr = img_array.ctypes.data_as(POINTER(c_ubyte))

            # Apply flags
            flags = 0
            if bottomup:
                flags |= TJFLAG.BOTTOMUP.value
            if fast_upsample:
                flags |= TJFLAG.FASTUPSAMPLE.value
            if fast_dct:
                flags |= TJFLAG.FASTDCT.value
            if accurate_dct:
                flags |= TJFLAG.ACCURATEDCT.value

            # Decompress
            status = self.__decompress(
                handle,
                src_addr, jpeg_array.size,
                dest_addr,
                scaled_width, 0, scaled_height,
                pixel_format.value, c_int(flags)
            )
            if status != 0:
                raise IOError(self.__get_error_str().decode())

            # Done!
            return img_array

    # --- Encoding

    def encode(self,
               img_array,
               quality=85,
               pixel_format=TJPF.BGR,
               jpeg_subsample=TJSAMP.YUV422,
               progressive=False,
               bottomup=False,
               norealloc=False,
               fast_dct=True,
               accurate_dct=False):
        """Encode numpy array to JPEG buffer."""
        handle = self.__init_compress()
        try:
            # Result vars
            jpeg_buf = c_void_p()
            jpeg_size = c_ulong()
            # Source params
            height, width, _ = img_array.shape
            src_addr = img_array.ctypes.data_as(POINTER(c_ubyte))
            # Apply flags
            flags = 0
            if bottomup:
                flags |= TJFLAG.BOTTOMUP.value
            if norealloc:
                flags |= TJFLAG.NOREALLOC.value
            if progressive:
                flags |= TJFLAG.PROGRESSIVE.value
            if fast_dct:
                flags |= TJFLAG.FASTDCT.value
            if accurate_dct:
                flags |= TJFLAG.ACCURATEDCT.value
            # Compress
            status = self.__compress(
                handle, src_addr,
                width, 0, height,
                pixel_format.value,
                byref(jpeg_buf), byref(jpeg_size),
                jpeg_subsample.value,
                quality,
                flags
            )
            if status != 0:
                raise IOError(self.__get_error_str().decode())
            # Trim the buffer to keep only needed memory
            # TODO: we should allow here to pass the dest buffer already, avoid memmoves & friends
            dest_buf = create_string_buffer(jpeg_size.value)
            memmove(dest_buf, jpeg_buf.value, jpeg_size.value)
            self.__free(jpeg_buf)
            return dest_buf.raw
        finally:
            self.__destroy(handle)

    # --- Misc

    @staticmethod
    def conda_turbo_jpeg_so(prefix=None):
        if not prefix:
            prefix = os.environ.get('CONDA_PREFIX')

        if not prefix:
            raise Exception('No prefix found, are we running under a conda environment?')

        candidates = [
            # Our prefixed version
            op.join(prefix, 'lib', 'libjpeg-turbo', 'prefixed', 'lib', 'libturbojpeg.so'),
            # Our non-prefixed version
            op.join(prefix, 'lib', 'libjpeg-turbo', 'lib', 'libturbojpeg.so'),
            # Conda-forge's version
            op.join(prefix, 'lib', 'libturbojpeg.so'),
        ]
        for candidate in candidates:
            if op.exists(candidate):
                return op.realpath(candidate)
        raise Exception('No libturbojpeg found in prefix %r' % prefix)

    def _find_turbo(self, lib_path=None):
        # TODO: we could even just distribute this with the package ourselves...
        if lib_path is None:
            try:
                lib_path = self.conda_turbo_jpeg_so()
            except Exception:
                candidates = {
                    'Darwin':  [
                        '/usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib'
                    ],
                    'Linux':   [
                        '/usr/lib/libturbojpeg.so',
                        '/usr/local/lib/libturbojpeg.so',
                        '/opt/libjpeg-turbo/lib64/libturbojpeg.so'
                    ],
                    'Windows': [
                        'C:/libjpeg-turbo64/bin/turbojpeg.dll'
                    ]
                }.get(platform.system(), ())
                for candidate in candidates:
                    if op.exists(candidate):
                        lib_path = candidate
                        break
        if lib_path is None:
            lib_path = find_library('turbojpeg')
        if lib_path is None or not op.exists(lib_path):
            raise Exception('Cannot find libjpeg-turbo shared library')
        return op.realpath(lib_path)

    @property
    def turbo_version(self):
        if platform.system() != 'Linux':
            raise NotImplementedError('Not implemented for platform %r' % platform.system())
        return '.'.join(self.lib_path.split('.')[-3:])


#
# TODO: monkeypatch cv2 to use this if available.
#   - Find in asking for jpeg
#   - Reroute parameters as needed
# def patch_cv2():
#     import imghdr
#     cv2.imencode = None
#     cv2.imdecode = None
#

#
# TODO: expose more API, like:
#   - tjDecodeYUV, tjDecodeYUVPlanes,
#   - tjTransform
#     can crop be advantageous? if not, remember there is support for fast partial decoding
#     in libjpeg API (see, e.g., decode_and_crop_jpeg)
#   - expose libjpeg API when needed (e.g. for stuff not exposed by turbojpeg higher level API)
# PROTIP: open libjpeg.txt, turbojpeg.h, turbojpeg.c and have them always closeby...
#

#
# TODO: allow to specify output buffer for encode / decode
#
