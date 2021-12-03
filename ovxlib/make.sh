#!/bin/bash

CONFIG=""

if [ "$VSI_GPERF_DEBUG" = "1" ]; then
    pushd third-party/gperftools
    bash autogen.sh
    ./configure
    make -f Makefile
    popd
fi

if [ "$1" = "NNAPI0.3" ]; then
    CONFIG="VSI_CFG_NNAPI_0_3"
elif [ "$1" = "NNAPI0.4" ]; then
    CONFIG="VSI_CFG_NNAPI_0_4"
fi

bash gcc_gen_feature_config_header.sh .

echo $CONFIG
if [ "$CONFIG" != "" ]; then
    export OVXLIB_CONFIG=$CONFIG
    echo "OVXLIB_CONFIG = \"$OVXLIB_CONFIG\""
fi

rm ./lib/*
$cc clean
$cc install
