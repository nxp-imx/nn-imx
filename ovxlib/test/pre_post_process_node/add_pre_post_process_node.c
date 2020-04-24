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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vsi_nn_pub.h"

/*-------------------------------------------
                   Macros
 -------------------------------------------*/
#define _CHECK_TENSOR_ID( id, lbl )      do {\
    if( VSI_NN_TENSOR_ID_NA == id ) {\
        printf("CHECK TENSOR ID %d", __LINE__);\
        goto lbl;\
        }\
    } while(0)

#define _CHECK_PTR( ptr, lbl )      do {\
    if( NULL == ptr ) {\
        printf("CHECK PTR %d", __LINE__);\
        goto lbl;\
    }\
} while(0)

#define _CHECK_STATUS( stat, lbl )  do {\
    if( VX_SUCCESS != stat ) {\
        printf("CHECK STATUS %d", __LINE__);\
        goto lbl;\
    }\
} while(0)

#define _CONST_NODE_NUM     0
#define _TENSOR_NUM         (2+2)  /*origin input and output norm tensors and new input and
                                   output norm tensor added after adding preprocess and postprocess node */
#define _NODE_NUM           (1+2)  /*origin leakyrelu node and added two nodes */

/*-------------------------------------------
                  Variable definitions
-------------------------------------------*/
/*pre process for lid: leakyrelu*/
static vsi_nn_preprocess_source_layout_e source_layout_for_norm_tensor_0 = VSI_NN_SOURCE_LAYOUT_NCHW;
static vsi_nn_preprocess_source_format_e source_format_for_norm_tensor_0 = VSI_NN_SOURCE_FORMAT_TENSOR;
static int32_t perm_0[] = {0, 1, 2, 3};
static vsi_nn_preprocess_permute_t permute_for_norm_tensor_0 = {perm_0, 4};
static vsi_nn_preprocess_base_t pre_process_for_norm_tensor_0[] =
    {
    {VSI_NN_PREPROCESS_SOURCE_LAYOUT, &source_layout_for_norm_tensor_0},
    {VSI_NN_PREPROCESS_SET_SOURCE_FORMAT, &source_format_for_norm_tensor_0},
    {VSI_NN_PREPROCESS_PERMUTE, &permute_for_norm_tensor_0},
    };

/*post process for lid: leakyrelu*/
static int32_t perm_1[] = {0, 1, 2, 3};
static vsi_nn_postprocess_permute_t permute_for_norm_tensor_1 = {perm_1, 4};
static vsi_nn_postprocess_dtype_convert_t dtype_convert_for_norm_tensor_1 = {{VSI_NN_DIM_FMT_NCHW, VSI_NN_TYPE_FLOAT32, {VSI_NN_QNT_TYPE_NONE}}};
static vsi_nn_postprocess_base_t post_process_for_norm_tensor_1[] =
    {
    {VSI_NN_POSTPROCESS_PERMUTE, &permute_for_norm_tensor_1},
    {VSI_NN_POSTPROCESS_DTYPE_CONVERT, &dtype_convert_for_norm_tensor_1},
    };

/*-------------------------------------------
                    Functions
-------------------------------------------*/
static uint8_t *_get_tensor_data
    (
    vsi_nn_tensor_t *tensor,
    const char *filename
    )
{
    vsi_status status = VSI_FAILURE;
    uint32_t i = 0;
    float fval = 0.0;
    uint8_t *tensorData;
    uint32_t sz = 1;
    uint32_t stride = 1;
    FILE *tensorFile;

    tensorData = NULL;
    tensorFile = fopen(filename, "rb");
    TEST_CHECK_PTR(tensorFile, error);

    sz = vsi_nn_GetElementNum(tensor);
    stride = vsi_nn_TypeGetBytes(tensor->attr.dtype.vx_type);
    tensorData = (uint8_t *)malloc(stride * sz * sizeof(uint8_t));
    TEST_CHECK_PTR(tensorData, error);
    memset(tensorData, 0, stride * sz * sizeof(uint8_t));

    for(i = 0; i < sz; i++)
    {
        if(fscanf( tensorFile, "%f ", &fval ) != 1)
        {
            printf("Read tensor file fail.\n");
            printf("Please check file lines or if the file contains illegal characters\n");
            goto error;
        }
        status = vsi_nn_Float32ToDtype(fval, &tensorData[stride * i], &tensor->attr.dtype);
        TEST_CHECK_STATUS(status, error);
    }

    if(tensorFile)fclose(tensorFile);
    return tensorData;
error:
    if(tensorFile)fclose(tensorFile);
    return NULL;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char *argv[])
{
vx_status             status;
vsi_nn_graph_t      * graph;
vsi_nn_context_t      ctx;
vsi_nn_node_t       * node;
vsi_nn_tensor_attr_t  attr;
vsi_nn_tensor_id_t    input[1];
vsi_nn_tensor_id_t    output[1];
vsi_nn_tensor_t     * in_tensor;
vsi_nn_tensor_t     * out_tensor;
const char          *filename;
uint8_t             *tensorData = NULL;
char                dumpInput[128];
char                dumpOutput[128];

status = VSI_FAILURE;
filename = ((const char **)argv + 1)[0];

ctx = vsi_nn_CreateContext();
_CHECK_PTR(ctx, final);

graph = vsi_nn_CreateGraph(ctx, _TENSOR_NUM, _NODE_NUM);
_CHECK_PTR(graph, final);

vsi_nn_SetGraphInputs(graph, NULL, 1);
vsi_nn_SetGraphOutputs(graph, NULL, 1);

node = vsi_nn_AppendNode(graph, VSI_NN_OP_LEAKY_RELU, NULL);
node->uid = 0;
node->nn_param.activation.leaky_ratio = 0.1;
_CHECK_PTR( node, final );

memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
attr.size[0] = 224;
attr.size[1] = 224;
attr.size[2] = 3;
attr.size[3] = 1;
attr.dim_num = 4;
attr.vtl = FALSE;
attr.is_const = FALSE;
attr.dtype.vx_type = VSI_NN_TYPE_FLOAT16;
attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;

input[0] = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
output[0] = vsi_nn_AddTensor(graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL);
_CHECK_TENSOR_ID(input[0], final);
_CHECK_TENSOR_ID(output[0], final);

node->input.tensors[0] = input[0];
node->output.tensors[0] = output[0];

graph->input.tensors[0] = input[0];
graph->output.tensors[0] = output[0];

/* Add preprocess and postprocess node */
status = vsi_nn_AddGraphPreProcess(graph, 0, pre_process_for_norm_tensor_0, sizeof(pre_process_for_norm_tensor_0)/sizeof(vsi_nn_preprocess_base_t));
_CHECK_STATUS( status, final );
status = vsi_nn_AddGraphPostProcess(graph, 0, post_process_for_norm_tensor_1, sizeof(post_process_for_norm_tensor_1)/sizeof(vsi_nn_postprocess_base_t));
_CHECK_STATUS(status, final);

status = vsi_nn_SetupGraph(graph, TRUE);
_CHECK_STATUS(status, final);

/* copy data to input tensor */
in_tensor = vsi_nn_GetTensor(graph, graph->input.tensors[0]);
tensorData = _get_tensor_data(in_tensor,filename);
status = vsi_nn_CopyDataToTensor(graph, in_tensor, tensorData);
TEST_CHECK_STATUS(status, final);

/* Save input tensor data to file */
snprintf(dumpInput, sizeof(dumpInput), "input.txt");
vsi_nn_SaveTensorToTextByFp32(graph,in_tensor, dumpInput, NULL);

status = vsi_nn_VerifyGraph(graph);
_CHECK_STATUS(status, final);
status = vsi_nn_RunGraph(graph);
_CHECK_STATUS(status, final);

/* Save output tensor data to file */
out_tensor = vsi_nn_GetTensor(graph, graph->output.tensors[0]);
snprintf(dumpOutput, sizeof(dumpOutput), "output.txt");
vsi_nn_SaveTensorToTextByFp32(graph, out_tensor, dumpOutput, NULL);

final:

vsi_nn_ReleaseGraph(&graph);
vsi_nn_ReleaseContext(&ctx);

return status;
}


