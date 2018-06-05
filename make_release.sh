#!/usr/bin/bash

# Run from xsjsda23 at root of repo

# ./make_release <new_branch_name>
if [ "$#" -ne 1 ]; then
    echo "Please provide new public branch name"
    exit
fi

git checkout -b $1

git branch

echo "COMPILING C++"
cd xfdnn/rt/xdnn_cpp
make clean
make -j 10 XBLASOPTS="-O3 -D CBLAS -D XDNN_V2"
cd ../../../

echo "COMPILING PYTHON"
python -m compileall -f ./xfdnn/tools/compile
python -m compileall -f ./xfdnn/tools/quantize

echo "REMOVING RANDOM BACKUP FILE THAT SHOULDNT BE IN THE REPO"
git rm -rf ./xfdnn/tools/quantize/quantize_base.py.bak

echo "REMOVING XFDNN C++ and H++"
find ./xfdnn/rt/xdnn_cpp -maxdepth 1 -name "*.cpp" -exec git rm {} \;
find ./xfdnn/rt/xdnn_cpp -maxdepth 1 -name "*.h" -exec git rm {} \;

echo "ADDING XFDNN C++ Library"
git add ./xfdnn/rt/xdnn_cpp/lib/libxblas.so
git add ./xfdnn/rt/xdnn_cpp/lib/libxfdnn.so
git add ./xfdnn/rt/xdnn_cpp/objs/xblas.o
git add ./xfdnn/rt/xdnn_cpp/objs/xdnn.o
git add ./xfdnn/rt/xdnn_cpp/objs/xmlrt.o

echo "REMOVING COMPILER QUANTIZER PYTHON"
find ./xfdnn/tools/compile  -name "*.py" ! -name "__init__.py" -exec git rm {} \;
find ./xfdnn/tools/quantize -name "*.py" ! -name "__init__.py" -exec git rm {} \;

echo "ADDING COMPILER QUANTIZER COMPILED PYTHON"
find ./xfdnn/tools/compile  -name "*.pyc" -exec git add -f {} \;
find ./xfdnn/tools/quantize -name "*.pyc" -exec git add -f {} \;

echo "GIT COMMIT"
git commit -m "Created public branch"

echo "GIT PUSH"
git push origin $1
