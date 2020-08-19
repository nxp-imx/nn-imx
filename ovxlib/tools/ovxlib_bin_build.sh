#!/bin/bash

if [ -z $5 ]; then
echo
echo usage:
echo "    $0 AQROOT OVXLIB buildTool buildConfig GPU_CONFIG [clean]"
echo
echo "   AQROOT: vivante driver root path"
echo "   OVXLIB: ovxlib root path"
echo "   toolChain: x86_gcc|x86_x64_gcc"
echo "     x86_gcc: use native gcc to build 32bit offline compiler"
echo "     x86_x64_gcc: use native gcc to build 64bit offline compiler"
echo "   buildConfig: debug|release"
echo
echo "e.g."
echo "    ./ovxlib_bin_build.sh AQROOT OVXLIB x86_gcc release vip8000"
echo "    ./ovxlib_bin_build.sh AQROOT OVXLIB x86_x64_gcc release vip8000"
echo
exit 1
fi

export AQROOT=$1
export OVXLIB=$2
export BUILD_TOOL=$3
export BUILD_CONFIG=$4
export GPU_CONFIG=$5
export MORE=$6
export GPU_CONFIG_FILE=$AQROOT/compiler/vclcompiler/viv_gpu.config

function init()
{
    chmod +x ./build5x.sh
    cd $AQROOT
    if [ "$BUILD_TOOL" = "x86_x64_gcc" ]; then
        ./build5x.sh -lxts ${BUILD_CONFIG} XAQ2 X86_NO_KERNEL FBVDK > $AQROOT/setenv
    fi

    if [ "$BUILD_TOOL" = "x86_gcc" ]; then
        ./build5x.sh -lxts ${BUILD_CONFIG} XAQ2 i386_NO_KERNEL FBVDK > $AQROOT/setenv
    fi

    if [ ! -e "$AQROOT/setenv" ]; then
        echo "ERROR: not support this build tool: $BUILD_TOOL"
        exit 1
    fi

    echo "== check GPU config file ..."
    GPU_CONFIG_FILE=$AQROOT/compiler/vclcompiler/viv_gpu.config
    if [ "$GPU_CONFIG" = "default" ]; then
        GPU_CONFIG_FILE=$AQROOT/compiler/vclcompiler/viv_gpu.config
    else
        GPU_CONFIG_FILE=$AQROOT/compiler/vclcompiler/viv_${GPU_CONFIG}.config
    fi
    if [ ! -e $GPU_CONFIG_FILE ]; then
        echo "ERROR: missing GPU config file: $GPU_CONFIG_FILE"
        echo "You can get GPU config file from here: //SW/Rel5x/configs/*.config"
        exit 1
    fi

    echo "== found GPU config file: $GPU_CONFIG_FILE"
    echo
}

function check_vcCompiler()
{
    echo "== check vcCompiler tool ... "
    if [ ! -e "$AQROOT/vcCompiler" ]; then
        echo "== not found vcCompiler"
        echo "== build vcCompiler ..."
        cd $AQROOT
        . ./setenv
        export CFLAGS="-w ${CFLAGS}"
        $cc -j4 install || exit 1
        cd $AQROOT/compiler/vclcompiler/source
        $cc install || exit 1
        cp -fv $SDK_DIR/samples/vclcompiler/vcCompiler $AQROOT
    else
        echo "== found vcCompiler tool: $AQROOT/vcCompiler"
    fi
}

function cleanup()
{

   if [ -e $AQROOT/setenv ]; then
       cd $AQROOT
       . ./setenv
       cd $AQROOT/compiler/vclcompiler/source
       $cc clean
       cd $AQROOT
       $cc clean
       if [ -e "$AQROOT/sdk/include" ]; then
           rm -rf $AQROOT/sdk/include
       fi
       cd $AQROOT/driver/khronos/libOpenVX/kernelBinaries/nnvxcBinaries/nnvxc_kernels
       rm -f *.gcPGM *.vxgcSL
   fi
}

function convert_vxc_shader()
{
    echo "== convert VXC shader to header files ..."
    if [ ! -e "$AQROOT/sdk/include" ]; then
       cd $AQROOT/sdk; ln -s inc include
    fi
    cd $OVXLIB/src/libnnext/ops/vx
    VX_BIN_PATH=$OVXLIB/include/libnnext/vx_bin
    if [ ! -e "$VX_BIN_PATH" ]; then
       mkdir -p $VX_BIN_PATH
    fi

    rm -f *.gcPGM *.vxgcSL vxc_*.h $VX_BIN_PATH/*.h

    #echo python $AQROOT/tools/bin/ExactVXC.py -i $AQROOT/driver/khronos/libOpenVX/driver/src/gc_vx_layer.c
    #python $AQROOT/tools/bin/ExactVXC.py -i gc_vx_layer.c || exit 1

    #echo python $AQROOT/tools/bin/myExtract.py gc_vx_layer.c "$GPU_CONFIG" || exit 1
    #python $AQROOT/tools/bin/myExtract.py gc_vx_layer.c "$GPU_CONFIG" || exit 1

    echo "== generating $VX_BIN_PATH/vxc_binaries.h ..."

    for vxFile in `ls *.vx | sed "s/\.vx//"`
    do
        if [ "${vxFile}" != "vsi_nn_kernel_header" ]; then
            echo $AQROOT/vcCompiler -f${GPU_CONFIG_FILE} -O0 -allkernel -o${vxFile} -m vsi_nn_kernel_header.vx ${vxFile}.vx
            $AQROOT/vcCompiler -f${GPU_CONFIG_FILE} -O0 -allkernel -o${vxFile} -m vsi_nn_kernel_header.vx ${vxFile}.vx || exit 1
            echo "python $AQROOT/tools/bin/ConvertPGMToH.py -i ${vxFile}_all.gcPGM -o $VX_BIN_PATH/vxc_bin_${vxFile}.h"
            python $AQROOT/tools/bin/ConvertPGMToH.py -i ${vxFile}_all.gcPGM -o $VX_BIN_PATH/vxc_bin_${vxFile}.h || exit 1
        fi
    done

    (
cat<<EOF
/****************************************************************************
*
*    Copyright (c) 2020 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* WARNING! AUTO-GENERATED, DO NOT MODIFY MANUALLY */

#ifndef __VXC_BINARIES_H__
#define __VXC_BINARIES_H__

EOF
    )>$VX_BIN_PATH/vxc_binaries.h

    for vxFile in `ls *.vx | sed "s/\.vx//"`
    do
        if [ "${vxFile}" != "vsi_nn_kernel_header" ]; then
            echo "#include \"vxc_bin_${vxFile}.h\"" >> $VX_BIN_PATH/vxc_binaries.h
        fi
    done

    (
cat<<EOF

#ifndef _cnt_of_array
#define _cnt_of_array( arr )            (sizeof( arr )/sizeof( arr[0] ))
#endif

typedef struct _vsi_nn_vx_bin_resource_item_type
{
    char const* name;
    uint8_t const* data;
    uint32_t len;
} vsi_nn_vx_bin_resource_item_type;

const vsi_nn_vx_bin_resource_item_type vx_bin_resource_items[] =
{
EOF
    )>>$VX_BIN_PATH/vxc_binaries.h

    for vxFile in `ls *.vx | sed "s/\.vx//"`
    do
        vxFileUpper=`echo ${vxFile} | tr 'a-z' 'A-Z'`
        if [ "${vxFile}" != "vsi_nn_kernel_header" ]; then
            echo "    {\"${vxFile}\", vxcBin${vxFile}, VXC_BIN_${vxFileUpper}_LEN}," >> $VX_BIN_PATH/vxc_binaries.h
        fi
    done

    (
cat<<EOF
};

const int vx_bin_resource_items_cnt = _cnt_of_array(vx_bin_resource_items);

#endif
EOF
    )>>$VX_BIN_PATH/vxc_binaries.h

    echo

    #python $AQROOT/tools/bin/ExtractVXCBins.py $OVXLIB/src/libnnext/ops/vx
    echo "== convert VXC shader to header files: success!"
    echo "== convert VXC shader to header files: success!"
    echo "== convert VXC shader to header files: success!"
    exit 0
}

if [ -z $MORE ] || [ "$MORE" != "clean" ]; then
    init
    check_vcCompiler
    export VIVANTE_SDK_DIR=$AQROOT/sdk
    convert_vxc_shader
else
    cleanup
fi

exit 0
