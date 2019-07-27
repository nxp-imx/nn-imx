/****************************************************************************
*
*    Copyright 2012 - 2019 Vivante Corporation, Santa Clara, California.
*    All Rights Reserved.
*
*    Permission is hereby granted, free of charge, to any person obtaining
*    a copy of this software and associated documentation files (the
*    'Software'), to deal in the Software without restriction, including
*    without limitation the rights to use, copy, modify, merge, publish,
*    distribute, sub license, and/or sell copies of the Software, and to
*    permit persons to whom the Software is furnished to do so, subject
*    to the following conditions:
*
*    The above copyright notice and this permission notice (including the
*    next paragraph) shall be included in all copies or substantial
*    portions of the Software.
*
*    THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
*    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
*    IN NO EVENT SHALL VIVANTE AND/OR ITS SUPPLIERS BE LIABLE FOR ANY
*    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
*    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
*    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/


#ifndef _VSI_NN_BINARY_TREE_H
#define _VSI_NN_BINARY_TREE_H

#if defined(__cplusplus)
extern "C"{
#endif

typedef long long vsi_nn_binary_tree_key_t;

#define vsi_nn_BinaryTreeInitRoot(n) do{n = NULL;} while (0);

typedef struct _vsi_nn_binary_tree
{
    struct _vsi_nn_binary_tree * left;
    struct _vsi_nn_binary_tree * right;
    vsi_nn_binary_tree_key_t     key;
    void * data_ptr;
} vsi_nn_binary_tree_t;

OVXLIB_API void vsi_nn_BinaryTreeRemoveNode
    (
    vsi_nn_binary_tree_t ** root,
    vsi_nn_binary_tree_key_t key
    );

OVXLIB_API void vsi_nn_BinaryTreeNewNode
    (
    vsi_nn_binary_tree_t ** root,
    vsi_nn_binary_tree_key_t key,
    void * data
    );

OVXLIB_API void * vsi_nn_BinaryTreeGetNode
    (
    vsi_nn_binary_tree_t ** root,
    vsi_nn_binary_tree_key_t key
    );

#if defined(__cplusplus)
}
#endif

#endif
