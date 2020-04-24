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
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vsi_nn_pub.h"

#include "vnn_global.h"
#include "vnn_post_process.h"

#include <VX/vx_khr_cnn.h>
#define _BASETSD_H

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void vnn_SaveOutputData(vsi_nn_graph_t *graph)
{
vx_uint32 i,j;
vx_char filename[64] = {0}, tmp[8] = {0};
vsi_nn_tensor_t *tensor;

for(i = 0; i < graph->output.num; i++)
    {
    tensor = vsi_nn_GetTensor(graph, graph->output.tensors[i]);

    sprintf(filename, "output%u", i);
    for(j = 0; j < tensor->attr.dim_num; j++)
        {
        memset(tmp, 0, sizeof(tmp));
        sprintf(tmp, "_%u", tensor->attr.size[j]);
        strcat(filename, tmp);
        }
    strcat(filename, ".txt");

    vsi_nn_SaveTensorToTextByFp32( graph, tensor, filename, NULL );
    }
}

static vx_bool vnn_GetTop
    (
    vx_float32 *pfProb,
    vx_float32 *pfMaxProb,
    vx_uint32 *pMaxClass,
    vx_uint32 outputCount,
    vx_uint32 topNum
    )
{
    vx_uint32 i, j;

#define MAX_TOP_NUM 20
if (topNum > MAX_TOP_NUM) return vx_false_e;

memset(pfMaxProb, 0, sizeof(vx_float32) * topNum);
memset(pMaxClass, 0xff, sizeof(vx_float32) * topNum);

for (j = 0; j < topNum; j++)
    {
    for (i=0; i<outputCount; i++)
        {
        if ((i == *(pMaxClass+0)) || (i == *(pMaxClass+1)) || (i == *(pMaxClass+2)) ||
            (i == *(pMaxClass+3)) || (i == *(pMaxClass+4)))
            continue;

        if (pfProb[i] > *(pfMaxProb+j))
            {
            *(pfMaxProb+j) = pfProb[i];
            *(pMaxClass+j) = i;
            }
        }
    }

return vx_true_e;
}

static vx_status vnn_ShowTp5Result
    (
    vsi_nn_graph_t *graph,
    vsi_nn_tensor_t *tensor
    )
{
vx_status status = VX_FAILURE;
vx_uint32 i,sz;
vx_float32 *buffer = NULL;
vx_uint8 *tensor_data = NULL;
vx_uint32 MaxClass[5];
vx_float32 fMaxProb[5];

sz = 1;
for(i = 0; i < tensor->attr.dim_num; i++)
    {
    sz *= tensor->attr.size[i];
    }

tensor_data = (vx_uint8 *)vsi_nn_ConvertTensorToData(graph, tensor);
buffer = (vx_float32 *)malloc(sizeof(vx_float32) * sz);

for(i = 0; i < sz; i++)
    {
    status = vsi_nn_DtypeToFp32(tensor_data, &buffer[i], i, &tensor->attr.dtype);
    }

if (!vnn_GetTop(buffer, fMaxProb, MaxClass, sz, 5))
    {
    printf("Fail to show result.\n");
    goto final;
    }

printf(" --- Top5 ---\n");
for(i=0; i<5; i++)
    {
    printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
    }
status = VX_SUCCESS;

final:
if(tensor_data)free(tensor_data);
if(buffer)free(buffer);
return status;
}

vx_status vnn_PostProcessLenet(vsi_nn_graph_t *graph)
{
vx_status stauts = VX_FAILURE;

/* Show the top5 result */
stauts = vnn_ShowTp5Result(graph, vsi_nn_GetTensor(graph, graph->output.tensors[0]));
_CHECK_STATUS(stauts, final);

/* Save all output tensor data to txt file */
vnn_SaveOutputData(graph);

final:
return VX_SUCCESS;
}
