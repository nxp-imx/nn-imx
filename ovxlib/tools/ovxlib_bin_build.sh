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
    echo "== check GPU config file ..."
    GPU_CONFIG_FILE=$AQROOT/compiler/vclcompiler/viv_gpu.config
    if [ "$GPU_CONFIG" = "default" ]; then
        GPU_CONFIG_FILE=$AQROOT/compiler/vclcompiler/viv_gpu.config
    else
        GPU_CONFIG_FILE=$(ls $AQROOT/compiler/vclcompiler/*${GPU_CONFIG}.config)
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

        test -e "$AQARCH/reg" || exit $?
        cd $AQARCH/reg
        $cc || exit $?

        local depends=(hal/user hal/user/arch hal/os/linux/user compiler/libVSC compiler/libCLC)
        for p in ${depends[*]}
        do
            if [ -e "$AQROOT/$p" ]; then
                cd $AQROOT/$p
                $cc install || exit $?
            fi
        done

        cd $AQROOT/compiler/vclcompiler/source
        $cc install || exit 1
        cp -fv $SDK_DIR/samples/vclcompiler/vcCompiler $AQROOT
    else
        echo "== found vcCompiler tool: $AQROOT/vcCompiler"
    fi
}

function convert_vxc_shader()
{
    echo "== convert VXC shader to header files ..."
    if [ ! -e "$AQROOT/sdk/include" ]; then
       cd $AQROOT/sdk; ln -s inc include
    fi

    VX_BIN_PATH=$OVXLIB/include/libnnext/vx_bin
    if [ ! -e "$VX_BIN_PATH" ]; then
       mkdir -p $VX_BIN_PATH
    fi
    rm -f $VX_BIN_PATH/*.h

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

    cd $OVXLIB/src/libnnext/ops/vx
    rm -f *.gcPGM *.vxgcSL

    #echo python $AQROOT/tools/bin/ExactVXC.py -i $AQROOT/driver/khronos/libOpenVX/driver/src/gc_vx_layer.c
    #python $AQROOT/tools/bin/ExactVXC.py -i gc_vx_layer.c || exit 1

    #echo python $AQROOT/tools/bin/myExtract.py gc_vx_layer.c "$GPU_CONFIG" || exit 1
    #python $AQROOT/tools/bin/myExtract.py gc_vx_layer.c "$GPU_CONFIG" || exit 1

    echo "== generating $VX_BIN_PATH/vxc_binaries.h ..."

    for vxFile in `ls *.vx | sed "s/\.vx//"`
    do
    {
        if [ "${vxFile}" != "vsi_nn_kernel_header" ]; then
            echo $AQROOT/vcCompiler -f${GPU_CONFIG_FILE} -allkernel -cl-viv-gcsl-driver-image \
              -o${vxFile}_vx -m vsi_nn_kernel_header.vx ${vxFile}.vx
            $AQROOT/vcCompiler -f${GPU_CONFIG_FILE} -allkernel -cl-viv-gcsl-driver-image \
              -o${vxFile}_vx -m vsi_nn_kernel_header.vx ${vxFile}.vx || exit 1
            echo "python $AQROOT/tools/bin/ConvertPGMToH.py -i ${vxFile}_vx_all.gcPGM \
              -o $VX_BIN_PATH/vxc_bin_${vxFile}_vx.h"
            python $AQROOT/tools/bin/ConvertPGMToH.py -i ${vxFile}_vx_all.gcPGM \
              -o $VX_BIN_PATH/vxc_bin_${vxFile}_vx.h || exit 1
            echo "#include \"vxc_bin_${vxFile}_vx.h\"" >> $VX_BIN_PATH/vxc_binaries.h
        fi
    } &
    done

    wait

    python $AQROOT/tools/bin/ExtractVXCBins.py $OVXLIB/src/libnnext/ops/vx
    echo "== convert VXC shader to header files: success!"
    echo "== convert VXC shader to header files: success!"
    echo "== convert VXC shader to header files: success!"
    rm -f *.gcPGM *.vxgcSL _binary_interface.c _binary_interface.h

    cd $OVXLIB/src/libnnext/ops/cl
    rm -f *.gcPGM *.vxgcSL *.clgcSL

    for vxFile in `ls *.cl | sed "s/\.cl//"`
    do
    {
        if [ "${vxFile}" != "eltwise_ops_helper" ]; then
            cp ${vxFile}.cl ${vxFile}.vx
            echo $AQROOT/vcCompiler -f${GPU_CONFIG_FILE} -allkernel -cl-viv-gcsl-driver-image \
              -o${vxFile}_cl -m eltwise_ops_helper.cl ${vxFile}.vx
            $AQROOT/vcCompiler -f${GPU_CONFIG_FILE} -allkernel -cl-viv-gcsl-driver-image \
              -o${vxFile}_cl -m eltwise_ops_helper.cl ${vxFile}.vx || exit 1
            echo "python $AQROOT/tools/bin/ConvertPGMToH.py -i ${vxFile}_cl_all.gcPGM \
              -o $VX_BIN_PATH/vxc_bin_${vxFile}_cl.h"
            python $AQROOT/tools/bin/ConvertPGMToH.py -i ${vxFile}_cl_all.gcPGM \
              -o $VX_BIN_PATH/vxc_bin_${vxFile}_cl.h || exit 1
            echo "#include \"vxc_bin_${vxFile}_cl.h\"" >> $VX_BIN_PATH/vxc_binaries.h
            rm ${vxFile}.vx
        fi
    } &
    done

    wait

    python $AQROOT/tools/bin/ExtractVXCBins.py $OVXLIB/src/libnnext/ops/cl
    echo "== convert CL shader to header files: success!"
    echo "== convert CL shader to header files: success!"
    echo "== convert CL shader to header files: success!"
    rm -f *.gcPGM *.vxgcSL *.clgcSL _binary_interface.c _binary_interface.h

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

const vsi_nn_vx_bin_resource_item_type vx_bin_resource_items_vx[] =
{
EOF
    )>>$VX_BIN_PATH/vxc_binaries.h

    cd $OVXLIB/src/libnnext/ops/vx
    for vxFile in `ls *.vx | sed "s/\.vx//"`
    do
        vxFileUpper=`echo ${vxFile} | tr 'a-z' 'A-Z'`
        if [ "${vxFile}" != "vsi_nn_kernel_header" ]; then
            echo "    {\"${vxFile}_vx\", vxcBin${vxFile}_vx, VXC_BIN_${vxFileUpper}_VX_LEN}," \
            >> $VX_BIN_PATH/vxc_binaries.h
        fi
    done

    (
cat<<EOF
};

const int vx_bin_resource_items_vx_cnt = _cnt_of_array(vx_bin_resource_items_vx);

const vsi_nn_vx_bin_resource_item_type vx_bin_resource_items_cl[] =
{
EOF
    )>>$VX_BIN_PATH/vxc_binaries.h


    cd $OVXLIB/src/libnnext/ops/cl
    for vxFile in `ls *.cl | sed "s/\.cl//"`
    do
        vxFileUpper=`echo ${vxFile} | tr 'a-z' 'A-Z'`
        if [ "${vxFile}" != "eltwise_ops_helper" ]; then
            echo "    {\"${vxFile}_cl\", vxcBin${vxFile}_cl, VXC_BIN_${vxFileUpper}_CL_LEN}," \
            >> $VX_BIN_PATH/vxc_binaries.h
        fi
    done
    (
cat<<EOF
};

const int vx_bin_resource_items_cl_cnt = _cnt_of_array(vx_bin_resource_items_cl);

#endif
EOF
    )>>$VX_BIN_PATH/vxc_binaries.h

    exit 0
}

init
check_vcCompiler
export VIVANTE_SDK_DIR=$AQROOT/sdk
convert_vxc_shader

exit 0
