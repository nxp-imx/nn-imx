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

#include <VX/vx_khr_cnn.h>
#define _BASETSD_H

#include "vsi_nn_pub.h"

#include "vnn_global.h"
#include "vnn_pre_process.h"
#include "vnn_post_process.h"
#include "vnn_lenet.h"

/*-------------------------------------------
        Macros and Variables
-------------------------------------------*/

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static void vnn_ReleaseNeuralNetwork
    (
    vsi_nn_graph_t *graph
    )
{
vnn_ReleaseLenet( graph, vx_true_e );
}

static vx_status vnn_PostProcessNeuralNetwork
    (
    vsi_nn_graph_t *graph
    )
{
return vnn_PostProcessLenet( graph );
}

static vx_status vnn_ProcessGraph
    (
    vsi_nn_graph_t *graph
    )
{
vx_status status = VX_FAILURE;

/* Verify graph */
printf("Verify...\n");
status = vsi_nn_VerifyGraph( graph );
_CHECK_STATUS( status, final );

printf( "Start run graph...\n" );
status = vsi_nn_RunGraph( graph );
_CHECK_STATUS( status, final );

final:
return status;
}

static vx_status vnn_PrePocessNeuralNetwork
    (
    vsi_nn_graph_t *graph,
    vx_char *image_name
    )
{
return vnn_PrePocessLenet( graph, image_name );
}

static vsi_nn_graph_t *vnn_CreateNeuralNetwork
    (
    vx_char *data_file_name
    )
{
vsi_nn_graph_t *graph = NULL;

graph = vnn_CreateLenet(data_file_name, NULL);
_CHECK_PTR(graph, final);

/* Show the node and tensor */
vsi_nn_PrintGraph(graph);

final:
return graph;
}

/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main
    (
    int argc,
    char **argv
    )
{
vx_status status;
vsi_nn_graph_t *graph;
vx_char *data_name = NULL;
vx_char *image_name = NULL;

status = VX_FAILURE;
if(argc != 3)
    {
    printf("Usage:%s data_file_name image_file_name\n", argv[0]);
    return -1;
    }

data_name = (vx_char *)argv[1];
image_name = (vx_char *)argv[2];

/* Create the neural network */
graph = vnn_CreateNeuralNetwork( data_name );
_CHECK_PTR(graph, final);

/* Pre process the image data */
status = vnn_PrePocessNeuralNetwork( graph, image_name );
_CHECK_STATUS( status, final );

/* Verify and Process graph */
status = vnn_ProcessGraph( graph );
_CHECK_STATUS( status, final );

/* Post process output data */
status = vnn_PostProcessNeuralNetwork( graph );
_CHECK_STATUS( status, final );

final:
vnn_ReleaseNeuralNetwork( graph );
return status;
}

