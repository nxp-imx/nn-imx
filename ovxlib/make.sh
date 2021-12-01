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

#generate pre-build header to config features
echo Auto generate feature config header...
FEATURE_CONFIG_FILE="vsi_feature_config"
FEATURE_CONFIG_HEADER_FILE="./include/vsi_nn_feature_config.h"
if [ -f $FEATURE_CONFIG_HEADER_FILE ]; then
    rm -rf $FEATURE_CONFIG_HEADER_FILE
fi
echo "/****************************************************************************">> $FEATURE_CONFIG_HEADER_FILE
echo "*">> $FEATURE_CONFIG_HEADER_FILE
echo "*    Copyright (c) 2019 Vivante Corporation">> $FEATURE_CONFIG_HEADER_FILE
echo "*">> $FEATURE_CONFIG_HEADER_FILE
echo "*    Permission is hereby granted, free of charge, to any person obtaining a">> $FEATURE_CONFIG_HEADER_FILE
echo "*    copy of this software and associated documentation files (the "Software"),">> $FEATURE_CONFIG_HEADER_FILE
echo "*    to deal in the Software without restriction, including without limitation">> $FEATURE_CONFIG_HEADER_FILE
echo "*    the rights to use, copy, modify, merge, publish, distribute, sublicense,">> $FEATURE_CONFIG_HEADER_FILE
echo "*    and/or sell copies of the Software, and to permit persons to whom the">> $FEATURE_CONFIG_HEADER_FILE
echo "*    Software is furnished to do so, subject to the following conditions:">> $FEATURE_CONFIG_HEADER_FILE
echo "*">> $FEATURE_CONFIG_HEADER_FILE
echo "*    The above copyright notice and this permission notice shall be included in">> $FEATURE_CONFIG_HEADER_FILE
echo "*    all copies or substantial portions of the Software.">> $FEATURE_CONFIG_HEADER_FILE
echo "*">> $FEATURE_CONFIG_HEADER_FILE
echo "*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR">> $FEATURE_CONFIG_HEADER_FILE
echo "*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,">> $FEATURE_CONFIG_HEADER_FILE
echo "*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE">> $FEATURE_CONFIG_HEADER_FILE
echo "*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER">> $FEATURE_CONFIG_HEADER_FILE
echo "*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING">> $FEATURE_CONFIG_HEADER_FILE
echo "*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER">> $FEATURE_CONFIG_HEADER_FILE
echo "*    DEALINGS IN THE SOFTWARE.">> $FEATURE_CONFIG_HEADER_FILE
echo "*">> $FEATURE_CONFIG_HEADER_FILE
echo "*****************************************************************************/">> $FEATURE_CONFIG_HEADER_FILE
echo /*****Auto generated header file, Please DO NOT modify manually!*****/>> $FEATURE_CONFIG_HEADER_FILE
echo "#ifndef _VSI_NN_FEATURE_CONFIG_H">> $FEATURE_CONFIG_HEADER_FILE
echo "#define _VSI_NN_FEATURE_CONFIG_H">> $FEATURE_CONFIG_HEADER_FILE
echo "">> $FEATURE_CONFIG_HEADER_FILE
IFS_old=$IFS
IFS=$'\n'
for line in `cat $FEATURE_CONFIG_FILE`
do
    echo "$line">> $FEATURE_CONFIG_HEADER_FILE
done
IFS=$IFS_old
echo "">> $FEATURE_CONFIG_HEADER_FILE
echo "#endif">> $FEATURE_CONFIG_HEADER_FILE
echo "Generate feature config header to $FEATURE_CONFIG_HEADER_FILE successfully."

echo $CONFIG
if [ "$CONFIG" != "" ]; then
    export OVXLIB_CONFIG=$CONFIG
    echo "OVXLIB_CONFIG = \"$OVXLIB_CONFIG\""
fi

rm ./lib/*
$cc clean
$cc install
