#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

# If you call your environment something else, need to edit the below line

#parameters
ZERO=0
ONE=1
FAILED_STATUS=${ONE}
SUCCESSFUL_STATUS=${ZERO}
TRUE=${ONE}
FALSE=${ZERO}
DEBUG=${FALSE}

#parameters
envNameGlobal=
condaDirGlobal=

# USAGE
function usage
{
	local rtn=${SUCCESSFUL_EXIT_STATUS}
	echo "usage: $0 [ -e ENV_NAME || -h ]"
	echo "   ";
	echo "   This script enables caffe opencv symlinks for a conda environment."
	echo "   ";
	echo "  -e | --env               : Conda Environment name";
	echo "  -h | --help              : This message";
	return ${rtn}
}

# Execute
function execute
{
	local rtn=${SUCCESSFUL_EXIT_STATUS}
	cd ${condaDirGlobal}
	echo ${condaDirGlobal}

	ln -s libopencv_highgui.so libopencv_highgui.so.3.3
	ln -s libopencv_imgcodecs.so libopencv_imgcodecs.so.3.3
	ln -s libopencv_imgproc.so libopencv_imgproc.so.3.3
	ln -s libopencv_core.so libopencv_core.so.3.3

	cd -
	return ${rtn}
}

# Prereq
function prereq
{
	local rtn=${SUCCESSFUL_EXIT_STATUS}
	echo -ne "Checking prerequisites... "
	#conda is present 
	which conda &> /dev/null
	if [ ! $? -eq 0 ]; then
		echo "failed!"
		echo "conda is not present"
		rtn=${FAILED_EXIT_STATUS}
		exit ${FAILED_EXIT_STATUS}
		
	fi
	# conda directory
	local condaDir=`which conda`
	local suffix="/bin/conda"
	condaDir=${condaDir%"$suffix"}
	condaDirGlobal=${condaDir}/envs/${envNameGlobal}/lib
	if [ ! -d ${condaDirGlobal} ]
	then
		echo "failed!"
		echo "${condaDirGlobal} is not present"
		rtn=${FAILED_EXIT_STATUS}
		exit ${FAILED_EXIT_STATUS}
	fi
	echo "passed!"
	return ${rtn}
}

# Validate Arguments
function validate_args
{
	local rtn=${SUCCESSFUL_EXIT_STATUS}
	echo -ne "Validating Arguments... "
	echo "passed!"
	local envName="ml-suite"
	if [ ! -z "${envNameGlobal}" ] 
	then
		envName=${envNameGlobal}
	fi
	envNameGlobal=${envName}
	return ${rtn}
}

# Parse Arguments
function parse_args
{
	local rtn=${SUCCESSFUL_EXIT_STATUS}
	# positional args
	args=()

	# named args
	while [ "$1" != "" ]; do
		case "$1" in
			-e | --env )            shift
				                envNameGlobal=$1
				                ;;
			-h | --help )           usage
				                exit
				                ;;
			* )                     usage
				                exit 1
		esac
		shift
	done
	return ${rtn}
}

# Main
function main
{
	local rtn=${SUCCESSFUL_EXIT_STATUS}
	parse_args "$@"
	validate_args
	prereq
	execute
	return ${rtn}
}

main "$@";
