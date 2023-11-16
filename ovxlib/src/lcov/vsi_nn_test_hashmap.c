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
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "vsi_nn_test.h"
#include "vsi_nn_pub.h"
#include "lcov/vsi_nn_test_hashmap.h"
#define BUF_SIZE (64)

static vsi_status vsi_nn_test_hashmap_remove(void)
{
    vsi_nn_hashmap_t *map;
    map = (vsi_nn_hashmap_t *)malloc( sizeof(vsi_nn_hashmap_t) );
    if( NULL == map )
    {
        VSILOGE("Out of memory, create hashmap fail.");
    }
    memset( map, 0, sizeof( vsi_nn_hashmap_t ) );

    char hash_key[BUF_SIZE] = "a";
    char data[BUF_SIZE] = "a1";
    map->size = 0;
    vsi_nn_hashmap_item_t * iter;
    iter = (vsi_nn_hashmap_item_t *)vsi_nn_LinkListNewNode(sizeof( vsi_nn_hashmap_item_t ), NULL );
    uint8_t key_size = strlen( hash_key ) + 1;
    iter->hash_key = (char*)malloc( sizeof(char) * key_size );
    memcpy( iter->hash_key, hash_key, key_size );
    vsi_nn_LinkListPushStart( (vsi_nn_link_list_t **)&map->items, (vsi_nn_link_list_t *)iter );
    map->size = 1;
    iter->data = data;

    VSILOGI("vsi_nn_test_hashmap_get_size");
    const vsi_nn_hashmap_t *map1 = map;
    size_t result = vsi_nn_hashmap_get_size(map1);
    if(result != 1) return VSI_FAILURE;

    VSILOGI("vsi_nn_test_hashmap_remove");
    vsi_nn_hashmap_remove(map, hash_key);

    VSILOGI("vsi_nn_test_hashmap_get_size");
    result = vsi_nn_hashmap_get_size(map1);
    vsi_nn_hashmap_release(&map);
    if(0 == result)
    {
        return VSI_SUCCESS;
    }
    return VSI_FAILURE;
}

vsi_status vsi_nn_test_hashmap( void )
{
    vsi_status status = VSI_FAILURE;
    status = vsi_nn_test_hashmap_remove();
    TEST_CHECK_STATUS(status, final);

final:
    return status;
}