#!/bin/bash

if [ "$VSI_GPERF_DEBUG" = "1" ]; then
    pushd third-party/gperftools
    bash autogen.sh
    ./configure
    make -f Makefile
    popd
fi

bash gcc_gen_feature_config_header.sh .

rm ./lib/*
$cc clean
$cc install
