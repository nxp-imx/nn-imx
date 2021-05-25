#!/usr/bin/env python3
"""Build .vx/.cl files in src/libnnext/ops/vx(cl)/ to c file"""

from __future__ import absolute_import
from __future__ import print_function

import os
import fnmatch
import sys
from argparse import ArgumentParser

COPYRIGHT_TEXT = '''/****************************************************************************
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

VSI_NN_LIBNNEXT_VC_H_TEMPLATE = '''#COPYRIGHT#
/* WARNING! AUTO-GENERATED, DO NOT MODIFY MANUALLY */

#ifndef _VSI_NN_LIBNNEXT_RESOURCE_H
#define _VSI_NN_LIBNNEXT_RESOURCE_H

#include "kernel/vsi_nn_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Load gpu source code
 */
const char* vsi_nn_resource_load_source_code
    (
    const char* source_name,
    size_t* size,
    vsi_nn_kernel_type_e type
    );

#ifdef __cplusplus
}
#endif

#endif /* _VSI_NN_LIBNNEXT_RESOURCE_H */
'''

VSI_NN_LIBNNEXT_VC_C_TEMPLATE = '''#COPYRIGHT#
/* WARNING! AUTO-GENERATED, DO NOT MODIFY MANUALLY */

#include <stdlib.h>
#include <string.h>
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_util.h"
#include "kernel/vsi_nn_kernel.h"
#include "libnnext/vsi_nn_libnnext_resource.h"

#EVIS_FILE_CONTENT_ITEMS#

#CL_FILE_CONTENT_ITEMS#

typedef struct {
    char const* name;
    char const* data;
} source_map_t;

static const source_map_t evis_resource[] =
{#EVIS_RESOURCE_ITEMS#
};

static const source_map_t cl_resource[] =
{#CL_RESOURCE_ITEMS#
};

static const char* _load_code
    (
    const char* source_name,
    size_t* size,
    const source_map_t* source_map,
    size_t source_map_size,
    const char* tail
    )
{
    const char* source;
    char source_path[VSI_NN_MAX_PATH];
    size_t n;
    int i;
    source = NULL;
    n = snprintf( source_path, VSI_NN_MAX_PATH, "%s%s", source_name, tail );
    if( n == VSI_NN_MAX_PATH )
    {
        VSILOGE("Kernel source path overflow %d/%d", n, VSI_NN_MAX_PATH);
        *size = 0;
        return NULL;
    }
    for( i = 0; i < (int)source_map_size; i++ )
    {
        if( strncmp( source_map[i].name, source_path, VSI_NN_MAX_PATH ) == 0 )
        {
            source = source_map[i].data;
            *size = strlen( source );
            break;
        }
    }
    if( !source )
    {
        *size = 0;
    }
    return source;
} /* _load_code() */

const char* vsi_nn_resource_load_source_code
    (
    const char* source_name,
    size_t* size,
    vsi_nn_kernel_type_e type
    )
{
    const char* s = NULL;
    switch( type )
    {
        case VSI_NN_KERNEL_TYPE_EVIS:
            s = _load_code( source_name, size,
                evis_resource, _cnt_of_array(evis_resource), "_vx" );
            break;
        case VSI_NN_KERNEL_TYPE_CL:
            s = _load_code( source_name, size,
                cl_resource, _cnt_of_array(cl_resource), "_cl" );
            break;
        default:
            break;
    }
    return s;
} /* vsi_nn_resource_load_source_code() */
'''

VX_FILE_CONTETN_ITEM_TEMPLATE = '''\nstatic const char {name}[] = "{content}"; /* end of {name}*/\n'''

VX_RESOURCE_ITEM_TEMPLATE = '''\n    {{"{name}", {name}}},'''


def iterfindfiles(path, fnexp):
    for root, dirs, files in os.walk(path):
        for filename in fnmatch.filter(files, fnexp):
            yield os.path.join(root, filename)


def find_files_by_pattern(pattern, path='.'):
    paths = []
    for filename in iterfindfiles(path, pattern):
        if path not in paths:
            paths.append(filename)
    paths.sort()
    return paths

def load_resources(root, paths, pattern):
    file_content_items = ''
    resource_items = ''
    for source_path in paths:
        source_path = os.path.join(root, source_path)
        for path in find_files_by_pattern(pattern, path=source_path):
            with open(path) as fhndl:
                print('Processing {}'.format(path))
                content = fhndl.read().replace('\\','\\\\')\
                        .replace('\n', '\\n\\\n').replace('"', '\\"')
                variable_name = os.path.basename(path).replace('.', '_')
                file_content_items += VX_FILE_CONTETN_ITEM_TEMPLATE.format(name=variable_name, content=content)
                resource_items += VX_RESOURCE_ITEM_TEMPLATE.format(name=variable_name)
    return file_content_items,resource_items

def build_resource(root, evis_source_paths, cl_source_paths):
    output_c_file_path = os.path.join(root, 'src/libnnext/vsi_nn_libnnext_resource.c')
    output_h_file_path = os.path.join(root, 'include/libnnext/vsi_nn_libnnext_resource.h')
    evis = load_resources(root, evis_source_paths, "*.vx")
    cl   = load_resources(root, cl_source_paths, "*.cl")

    with open(output_h_file_path, 'w', newline='\n') as fhndl:
        fhndl.write(VSI_NN_LIBNNEXT_VC_H_TEMPLATE.replace('#COPYRIGHT#', COPYRIGHT_TEXT))

    with open(output_c_file_path, 'w', newline='\n') as fhndl:
        tmpl = VSI_NN_LIBNNEXT_VC_C_TEMPLATE.replace('#COPYRIGHT#', COPYRIGHT_TEXT)
        tmpl = tmpl.replace('#EVIS_FILE_CONTENT_ITEMS#', evis[0])\
                  .replace('#EVIS_RESOURCE_ITEMS#',     evis[1])\
                  .replace('#CL_FILE_CONTENT_ITEMS#', cl[0])\
                  .replace('#CL_RESOURCE_ITEMS#',     cl[1])
        fhndl.write(tmpl)

def main(argv):
    options = ArgumentParser(description='Build .vx file into c code')
    options.add_argument('--type',
                         default='embedded',
                         help='embedded: Only for embedded op\n'
                              'custom: Both embedded and custom op')
    args = options.parse_args()

    print('Build vx files'.center(45, '='))
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    evis_embedded_paths = ['src/libnnext/ops/vx/']
    evis_custom_paths = ['src/libnnext/ops/vx/',
                    'src/custom/ops/kernel/vx/']
    cl_embedded_paths = ['src/libnnext/ops/cl/']
    cl_custom_paths = ['src/libnnext/ops/cl/',
                    'src/custom/ops/kernel/cl/']

    if (args.type == 'custom'):
        build_resource(root_dir, evis_custom_paths, cl_custom_paths)
    else:
        build_resource(root_dir, evis_embedded_paths, cl_embedded_paths)
    print('=' * 45)

if __name__ == '__main__':
    main(sys.argv)
