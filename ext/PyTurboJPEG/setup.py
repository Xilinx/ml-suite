#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import io
from setuptools import setup, find_packages
setup(
    name='PyTurboJPEG',
    version='1.1.2',
    description='A Python wrapper of libjpeg-turbo for decoding and encoding JPEG image.',
    author='Lilo Huang',
    author_email='kuso.cc@gmail.com',
    url='https://github.com/lilohuang/PyTurboJPEG',
    license='MIT',
    install_requires=['numpy'],
    py_modules=['turbojpeg'],
    packages=find_packages(),
    long_description=io.open('README.md', encoding='utf-8').read()
)
