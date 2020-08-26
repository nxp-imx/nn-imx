#!/usr/bin/env python3
"""Add a embedded operation into ovxlib"""

from __future__ import absolute_import
from __future__ import print_function

import os
import fnmatch
import sys
from argparse import ArgumentParser

'''
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
'''

def get_blank_num(op, total_num):
    res = total_num - len(op)
    if res < 1:
        res = 1
    return res

def modify_file(modify_list, env):
    for m in modify_list:
        file = env.path + m['file']
        print("modify "+ file)
        with open(file, mode='r+', newline='\n', encoding='UTF-8') as fhndl:
            lines = m['func'](fhndl.readlines(), env)
            fhndl.seek(0)
            fhndl.writelines(lines)

def replace_keywords(op, new_lines, env):
    op = env.prefix + op
    KERNEL_OP_LOWER = op.lower()
    KERNEL_OP = op.upper()
    KERNEL_OP_STD = op[0].upper() + op[1:].lower()
    for index, line in enumerate(new_lines):
        if line.find('#KERNEL_OP#') != -1:
            new_lines[index] = new_lines[index].replace('#KERNEL_OP#', KERNEL_OP)
        if line.find('#KERNEL_OP_LOWER#') != -1:
            new_lines[index] = new_lines[index].replace('#KERNEL_OP_LOWER#', KERNEL_OP_LOWER)
        if line.find('#KERNEL_OP_STD#') != -1:
            new_lines[index] = new_lines[index].replace('#KERNEL_OP_STD#', KERNEL_OP_STD)
    return new_lines

def add_file(add_list, env):
    for a in add_list:
        file = env.path + a['file']
        with open(file, mode='r', newline='\n', encoding='UTF-8') as fhndl:
            lines = fhndl.readlines()
        for op in env.op_name:
            new_file = a['new_file'](op, env)
            print("add " + new_file)
            new_lines = lines.copy()
            with open(new_file, mode='w', newline='\n', encoding='UTF-8') as fhndl:
                new_lines = replace_keywords(op, new_lines, env)
                fhndl.writelines(new_lines)

def modify_BUILD(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('include/client/vsi_nn_vxkernel.h') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '        "include/ops/vsi_nn_op_' +
                             op.lower() + '.h",\n')
                offset += 1
        if line.find('src/libnnext/ops/kernel/vsi_nn_kernel_argmax.c') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '        "src/ops/vsi_nn_op_' +
                             op.lower() + '.c",\n')
                offset += 1
        if (not env.without_kernel) and line.find('src/libnnext/vsi_nn_libnnext_vx.c') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '        "src/libnnext/ops/kernel/vsi_nn_kernel_' +
                                 op.lower() + '.c",\n')
                offset += 1
    return new_lines

def modify_makefile_linux(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if (not env.without_kernel) and line.find('vsi_nn_kernel_tensorstackconcat.o') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '        $(OBJ_DIR)/vsi_nn_kernel_' +
                             op.lower() + '.o \\\n')
                offset += 1
        if line.find('vsi_nn_op_tensorstackconcat.o') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '             $(OBJ_DIR)/vsi_nn_op_' +
                                 op.lower() + '.o \\\n')
                offset += 1
    return new_lines

def modify_ovxlib_vcxproj(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('vsi_nn_op_tensorstackconcat.h') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    <ClInclude Include="include\\ops\\vsi_nn_op_' +
                             op.lower() + '.h" />\n')
                offset += 1
        if (not env.without_kernel) and line.find('vsi_nn_libnnext_vx.c') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    <ClCompile Include="src\\libnnext\\ops' +
                                 '\\kernel\\vsi_nn_kernel_' +
                                 op.lower() + '.c" />\n')
                offset += 1
        if line.find('vsi_nn_op_variable.c') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    <ClCompile Include="src\\ops\\vsi_nn_op_' +
                                 op.lower() + '.c" />\n')
                offset += 1
    return new_lines

def modify_ovxlib_vcxproj_filters(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('vsi_nn_vxkernel.h') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    <ClInclude Include="include\\ops\\vsi_nn_op_' +
                             op.lower() + '.h">\n      <Filter>Header Files' +
                             '\\ops</Filter>\n    </ClInclude>\n')
                offset += 1
        if line.find('vsi_nn_vxkernel.c') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    <ClCompile Include="src\\ops\\vsi_nn_op_' +
                                 op.lower() + '.c">\n      <Filter>Source Files' +
                                 '\\ops</Filter>\n    </ClCompile>\n')
                offset += 1
        if (not env.without_kernel) and line.find('vsi_pycc_interface.c') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    <ClCompile Include="src\\libnnext' +
                                 '\\ops\\kernel\\vsi_nn_kernel_' +
                                 op.lower() + '.c">\n      <Filter>Source Files\\libnnext' +
                                 '\\ops</Filter>\n    </ClCompile>\n')
                offset += 1
    return new_lines

def modify_vsi_nn_node_type_h(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('/* custom node head define define */') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset - 1, '#include "ops/vsi_nn_op_' +
                             op.lower() + '.h"\n')
                offset += 1
        if line.find('client_param[128]') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    vsi_nn_' +
                                 op.lower() + '_param' + ' ' * (get_blank_num(op, 19)) +
                                 op.lower() + ';\n')
                offset += 1
    return new_lines

def modify_vsi_nn_code_generator_c(lines, env):
    new_lines = lines.copy()
    offset = 0
    flag = False
    for index, line in enumerate(lines):
        if not flag and line.find('s_op_gen[]') != -1:
            flag = True
        elif flag and line.find('};') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    /* ' +
                                 op.upper() + ' */' + ' ' * (get_blank_num(op, 22)) +
                                 'NULL,\n')
                offset += 1
            flag = False
    return new_lines

def modify_vsi_nn_node_attr_template_c(lines, env):
    new_lines = lines.copy()
    offset = 0
    flag = False
    for index, line in enumerate(lines):
        if not flag and line.find('s_template[]') != -1:
            flag = True
        elif flag and line.find('};') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    /* ' +
                                 op.upper() + ' */' + ' ' * (get_blank_num(op, 22)) +
                                 'NULL,\n')
                offset += 1
            flag = False
    return new_lines

def modify_vx_lib_nnext_h(lines, env):
    new_lines = lines.copy()
    offset = 0
    flag = False
    for index, line in enumerate(lines):
        if not flag and line.find('vx_kernel_libnnext_offset_e') != -1:
            flag = True
        elif flag and line.find('};') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '    KERNEL_ENUM_' +
                                 op.upper() + ',\n')
                offset += 1
            flag = False
        if line.find('The Example Library Set') != -1:
            for op in env.op_name:
                op_std = op[0].upper() + op[1:].lower()
                new_lines.insert(index - 1 + offset, '#define VX_KERNEL_NAME_' +
                                 op.upper() + ' ' * (get_blank_num(op, 36)) +
                                 '"com.vivantecorp.extension.vxc' + op_std + '"\n')
                offset += 1
        if line.find('up to 0xFFF') != -1:
            for op in env.op_name:
                op_std = op[0].upper() + op[1:].lower()
                new_lines.insert(index + offset, '    VX_KERNEL_ENUM_' +
                                 op.upper() + ' ' * (get_blank_num(op, 21)) +
                                 '= VX_KERNEL_BASE(VX_ID_DEFAULT, VX_LIBRARY_LIBNNEXT) + KERNEL_ENUM_'
                                 + op.upper() + ',\n')
                offset += 1
    return new_lines

def modify_Android_mk(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if (not env.without_kernel) and line.find('vsi_nn_libnnext_vx.c') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '        libnnext/ops/kernel/vsi_nn_kernel_' +
                                 op.lower() + '.c \\\n')
                offset += 1
        if line.find('vsi_nn_op_lrn2.c') != -1:
            for op in env.op_name:
                new_lines.insert(index + offset, '             ops/vsi_nn_op_' +
                                 op.lower() + '.c \\\n')
                offset += 1
    return new_lines

def modify_custom_ops_def(lines, env):
    for op in env.op_name:
        lines.append('DEF_OP(' + env.prefix.upper() + op.upper() + ')\n')
    return lines

def modify_custom__node_type_def(lines, env):
    for op in env.op_name:
        lines.append('DEF_NODE_TYPE(' + env.prefix.lower() + op.lower() + ')\n')
    return lines

def modify_vsi_nn_custom_node_type_h(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('#endif') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset - 1, '#include "custom/ops/vsi_nn_op_' +
                             op.lower() + '.h"\n')
                offset += 1
    return new_lines

def modify_custom_BUILD(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('include/custom/vsi_nn_custom_node_type.h') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset + 1, '        "include/custom/ops/vsi_nn_op_' +
                             op.lower() + '.h",\n')
                offset += 1
        if line.find('src/custom/ops/vsi_nn_op_custom_softmax.c') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, '        "src/custom/ops/vsi_nn_op_' +
                             op.lower() + '.c",\n')
                offset += 1
        if (not env.without_kernel) and line.find('src/custom/ops/kernel/vsi_nn_kernel_custom_softmax.c') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, '        "src/custom/ops/kernel/vsi_nn_kernel_' +
                                 op.lower() + '.c",\n')
                offset += 1
    return new_lines

def modify_custom_makefile_linux(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('vsi_nn_op_custom_softmax.o') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset + 1, 'OBJECTS += $(OBJ_DIR)/vsi_nn_op_' +
                             op.lower() + '.o\n')
                offset += 1
        if (not env.without_kernel) and line.find('vsi_nn_kernel_custom_softmax.o') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset + 1, 'OBJECTS += $(OBJ_DIR)/vsi_nn_kernel_' +
                                 op.lower() + '.o\n')
                offset += 1
    return new_lines

def modify_custom_ovxlib_vcxproj(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('vsi_nn_op_custom_softmax.h') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, '    <ClInclude Include="include\\'
                             + 'custom\\ops\\vsi_nn_op_' +
                             op.lower() + '.h" />\n')
                offset += 1
        if (not env.without_kernel) and line.find('vsi_nn_kernel_custom_softmax.c') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, '    <ClCompile Include="src\\custom'
                                + '\\ops\\kernel\\vsi_nn_kernel_' +
                                op.lower() + '.c" />\n')
                offset += 1
        if line.find('vsi_nn_op_custom_softmax.c') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, '    <ClCompile Include="src\\custom'
                                 + '\\ops\\vsi_nn_op_' +
                                 op.lower() + '.c" />\n')
                offset += 1
    return new_lines

def modify_custom_ovxlib_vcxproj_filters(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('vsi_nn_op_custom_softmax.h') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, '    <ClInclude Include="include\\custom'
                             + '\\ops\\vsi_nn_op_' + op.lower() + '.h">\n      <Filter>'
                             + 'Header Files\\custom\\ops</Filter>\n    </ClInclude>\n')
                offset += 1
        if line.find('vsi_nn_op_custom_softmax.c') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, '    <ClCompile Include="src\\custom'
                                 + '\\ops\\vsi_nn_op_' +
                                 op.lower() + '.c">\n      <Filter>Source Files' +
                                 '\\custom\\ops</Filter>\n    </ClCompile>\n')
                offset += 1
        if (not env.without_kernel) and line.find('vsi_nn_kernel_custom_softmax.c') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, '    <ClCompile Include="src\\custom' +
                                 '\\ops\\kernel\\vsi_nn_kernel_' +
                                 op.lower() + '.c">\n      <Filter>Source Files\\custom' +
                                 '\\ops\\kernel</Filter>\n    </ClCompile>\n')
                offset += 1
    return new_lines

def modify_custom_Android_mk(lines, env):
    new_lines = lines.copy()
    offset = 0
    for index, line in enumerate(lines):
        if line.find('vsi_nn_op_custom_softmax.o') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, 'LOCAL_SRC_FILES += custom/ops/'
                                 + 'vsi_nn_op_' + op.lower() + '.c\n')
                offset += 1
        if (not env.without_kernel) and line.find('vsi_nn_kernel_custom_softmax.o') != -1:
            for op in env.op_name:
                op = env.prefix + op
                new_lines.insert(index + offset, 'LOCAL_SRC_FILES += custom/ops/kernel'
                                 + '/vsi_nn_kernel_' + op.lower() + '.c\n')
                offset += 1
    return new_lines

def add_vsi_nn_op_xxx_h_get_name(op, env):
    return env.path + '/include/ops/vsi_nn_op_' + op.lower() + '.h'

def add_vsi_nn_op_xxx_c_get_name(op, env):
    return env.path + '/src/ops/vsi_nn_op_' + op.lower() + '.c'

def add_vsi_nn_kernel_xxx_c_get_name(op, env):
    return env.path + '/src/libnnext/ops/kernel/vsi_nn_kernel_' + op.lower() + '.c'

def add_vsi_nn_kernel_xxx_vx_get_name(op, env):
    return env.path + '/src/libnnext/ops/vx/vsi_nn_kernel_' + op.lower() + '.vx'

def add_vsi_nn_op_custom_xxx_h_get_name(op, env):
    op = env.prefix + op
    return env.path + '/include/custom/ops/vsi_nn_op_' + op.lower() + '.h'

def add_vsi_nn_op_custom_xxx_c_get_name(op, env):
    op = env.prefix + op
    return env.path + '/src/custom/ops/vsi_nn_op_' + op.lower() + '.c'

def add_vsi_nn_kernel_custom_xxx_c_get_name(op, env):
    op = env.prefix + op
    return env.path + '/src/custom/ops/kernel/vsi_nn_kernel_' + op.lower() + '.c'

def add_vsi_nn_kernel_custom_xxx_vx_get_name(op, env):
    op = env.prefix + op
    return env.path + '/src/custom/ops/vx/vsi_nn_kernel_' + op.lower() + '.vx'

def modify_ops_def(lines, env):
    for op in env.op_name:
        lines.append('DEF_OP(' + op.upper() + ')\n')
    return lines

modify_list_for_embedded = [
    {'file': '/include/interface/ops.def', 'func': modify_ops_def},
    #{'file': '/BUILD', 'func': modify_BUILD},
    #{'file': '/src/makefile.linux', 'func': modify_makefile_linux},
    {'file': '/ovxlib.vcxproj', 'func': modify_ovxlib_vcxproj},
    {'file': '/ovxlib.vcxproj.filters', 'func': modify_ovxlib_vcxproj_filters},
    {'file': '/include/vsi_nn_node_type.h', 'func': modify_vsi_nn_node_type_h},
    {'file': '/src/utils/vsi_nn_code_generator.c', 'func': modify_vsi_nn_code_generator_c},
    {'file': '/src/vsi_nn_node_attr_template.c', 'func': modify_vsi_nn_node_attr_template_c},
    #{'file': '/include/libnnext/vx_lib_nnext.h', 'func': modify_vx_lib_nnext_h},
    #{'file': '/src/Android.mk', 'func': modify_Android_mk},
]

add_list_for_embedded = [
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.h',
     'new_file': add_vsi_nn_op_xxx_h_get_name},
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.c',
     'new_file': add_vsi_nn_op_xxx_c_get_name},
    {'file': '/tools/op_template/vsi_nn_kernel_#CLIENT_KERNEL#.c',
     'new_file': add_vsi_nn_kernel_xxx_c_get_name},
    {'file': '/tools/op_template/vsi_nn_kernel_#CLIENT_KERNEL#.vx',
     'new_file': add_vsi_nn_kernel_xxx_vx_get_name},
]

modify_list_for_embedded_without_kernel = [
    {'file': '/include/interface/ops.def', 'func': modify_ops_def},
    #{'file': '/BUILD', 'func': modify_BUILD},
    #{'file': '/src/makefile.linux', 'func': modify_makefile_linux},
    {'file': '/ovxlib.vcxproj', 'func': modify_ovxlib_vcxproj},
    {'file': '/ovxlib.vcxproj.filters', 'func': modify_ovxlib_vcxproj_filters},
    {'file': '/include/vsi_nn_node_type.h', 'func': modify_vsi_nn_node_type_h},
    {'file': '/src/utils/vsi_nn_code_generator.c', 'func': modify_vsi_nn_code_generator_c},
    {'file': '/src/vsi_nn_node_attr_template.c', 'func': modify_vsi_nn_node_attr_template_c},
    #{'file': '/src/Android.mk', 'func': modify_Android_mk},
]

add_list_for_embedded_without_kernel = [
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.h',
     'new_file': add_vsi_nn_op_xxx_h_get_name},
    {'file': '/tools/op_template/vsi_nn_op_#NO_KERNEL#.c',
     'new_file': add_vsi_nn_op_xxx_c_get_name},
]

modify_list_for_custom = [
    {'file': '/include/custom/custom_ops.def', 'func': modify_custom_ops_def},
    {'file': '/include/custom/custom_node_type.def', 'func': modify_custom__node_type_def},
    {'file': '/include/custom/vsi_nn_custom_node_type.h', 'func': modify_vsi_nn_custom_node_type_h},
    #{'file': '/BUILD', 'func': modify_custom_BUILD},
    #{'file': '/src/makefile.linux', 'func': modify_custom_makefile_linux},
    {'file': '/ovxlib.vcxproj', 'func': modify_custom_ovxlib_vcxproj},
    {'file': '/ovxlib.vcxproj.filters', 'func': modify_custom_ovxlib_vcxproj_filters},
    #{'file': '/src/Android.mk', 'func': modify_custom_Android_mk},
]

add_list_for_custom = [
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.h',
     'new_file': add_vsi_nn_op_custom_xxx_h_get_name},
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.c',
     'new_file': add_vsi_nn_op_custom_xxx_c_get_name},
    {'file': '/tools/op_template/vsi_nn_kernel_#CUSTOM_KERNEL#.c',
     'new_file': add_vsi_nn_kernel_custom_xxx_c_get_name},
    {'file': '/tools/op_template/vsi_nn_kernel_#CLIENT_KERNEL#.vx',
     'new_file': add_vsi_nn_kernel_custom_xxx_vx_get_name},
]

modify_list_for_custom_without_kernel = [
    {'file': '/include/custom/custom_ops.def', 'func': modify_custom_ops_def},
    {'file': '/include/custom/custom_node_type.def', 'func': modify_custom__node_type_def},
    {'file': '/include/custom/vsi_nn_custom_node_type.h', 'func': modify_vsi_nn_custom_node_type_h},
    #{'file': '/BUILD', 'func': modify_custom_BUILD},
    #{'file': '/src/makefile.linux', 'func': modify_custom_makefile_linux},
    {'file': '/ovxlib.vcxproj', 'func': modify_custom_ovxlib_vcxproj},
    {'file': '/ovxlib.vcxproj.filters', 'func': modify_custom_ovxlib_vcxproj_filters},
    #{'file': '/src/Android.mk', 'func': modify_custom_Android_mk},
]

add_list_for_custom_without_kernel = [
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.h',
     'new_file': add_vsi_nn_op_custom_xxx_h_get_name},
    {'file': '/tools/op_template/vsi_nn_op_#NO_KERNEL#.c',
     'new_file': add_vsi_nn_op_custom_xxx_c_get_name},
]

modify_list_for_internal = [
    {'file': '/include/internal/internal_ops.def', 'func': modify_ops_def},
    #{'file': '/BUILD', 'func': modify_BUILD},
    #{'file': '/src/makefile.linux', 'func': modify_makefile_linux},
    {'file': '/ovxlib.vcxproj', 'func': modify_ovxlib_vcxproj},
    {'file': '/ovxlib.vcxproj.filters', 'func': modify_ovxlib_vcxproj_filters},
    {'file': '/include/vsi_nn_node_type.h', 'func': modify_vsi_nn_node_type_h},
    #{'file': '/include/libnnext/vx_lib_nnext.h', 'func': modify_vx_lib_nnext_h},
    #{'file': '/src/Android.mk', 'func': modify_Android_mk},
]

add_list_for_internal = [
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.h',
     'new_file': add_vsi_nn_op_xxx_h_get_name},
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.c',
     'new_file': add_vsi_nn_op_xxx_c_get_name},
    {'file': '/tools/op_template/vsi_nn_kernel_#CLIENT_KERNEL#.c',
     'new_file': add_vsi_nn_kernel_xxx_c_get_name},
    {'file': '/tools/op_template/vsi_nn_kernel_#CLIENT_KERNEL#.vx',
     'new_file': add_vsi_nn_kernel_xxx_vx_get_name},
]

modify_list_for_internal_without_kernel = [
    {'file': '/include/internal/internal_ops.def', 'func': modify_ops_def},
    #{'file': '/BUILD', 'func': modify_BUILD},
    #{'file': '/src/makefile.linux', 'func': modify_makefile_linux},
    {'file': '/ovxlib.vcxproj', 'func': modify_ovxlib_vcxproj},
    {'file': '/ovxlib.vcxproj.filters', 'func': modify_ovxlib_vcxproj_filters},
    {'file': '/include/vsi_nn_node_type.h', 'func': modify_vsi_nn_node_type_h},
    #{'file': '/src/Android.mk', 'func': modify_Android_mk},
]

add_list_for_internal_without_kernel = [
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.h',
     'new_file': add_vsi_nn_op_xxx_h_get_name},
    {'file': '/tools/op_template/vsi_nn_op_#CLIENT_KERNEL#.c',
     'new_file': add_vsi_nn_op_xxx_c_get_name},
]

class Env():
    pass

def main(argv):
    options = ArgumentParser(description='Add a embedded operation into ovxlib')
    options.add_argument('--type',
                         default='embedded',
                         help='Embedded/Custom/Internal')
    options.add_argument('--without-kernel',
                         action='store_true',
                         help='Without kernel for op')
    options.add_argument('--prefix',
                         default='custom',
                         help='The prefix for custom op')
    options.add_argument('op_name', nargs = '*')
    args = options.parse_args()

    env = Env()
    env.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env.op_name = args.op_name
    env.without_kernel = args.without_kernel
    if (args.type == 'custom'):
        env.prefix = args.prefix + '_'
        if args.without_kernel:
            modify_file(modify_list_for_custom_without_kernel, env)
            add_file(add_list_for_custom_without_kernel, env)
        else:
            modify_file(modify_list_for_custom, env)
            add_file(add_list_for_custom, env)
    elif (args.type == 'internal'):
        env.prefix = ''
        if args.without_kernel:
            modify_file(modify_list_for_internal_without_kernel, env)
            add_file(add_list_for_internal_without_kernel, env)
        else:
            modify_file(modify_list_for_internal, env)
            add_file(add_list_for_internal, env)
    else:
        env.prefix = ''
        if args.without_kernel:
            modify_file(modify_list_for_embedded_without_kernel, env)
            add_file(add_list_for_embedded_without_kernel, env)
        else:
            modify_file(modify_list_for_embedded, env)
            add_file(add_list_for_embedded, env)

    print('=' * 45)

if __name__ == '__main__':
    main(sys.argv)
