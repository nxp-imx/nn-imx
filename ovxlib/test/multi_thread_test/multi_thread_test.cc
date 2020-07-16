#include <assert.h>
#include <thread>
#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <memory>
#include "vsi_nn_pub.h"

#define VNN_APP_DEBUG 0

#define NEW_VXNODE(_node, _type, _in, _out, _uid) do {\
        _node = vsi_nn_AddNode( graph, _type, _in, _out, NULL );\
        _node->uid = (uint32_t)_uid; \
        if( NULL == _node ) {\
            goto error;\
        }\
    } while(0)

#define NEW_VIRTUAL_TENSOR(_id, _attr, _dtype) do {\
        memset( _attr.size, 0, VSI_NN_MAX_DIM_NUM * sizeof(uint32_t));\
        _attr.dim_num = VSI_NN_DIM_AUTO;\
        _attr.vtl = !VNN_APP_DEBUG;\
        _attr.is_const = FALSE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, NULL );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

// Set const tensor dims out of this macro.
#define NEW_CONST_TENSOR(_id, _attr, _dtype, _ofst, _size) do {\
        data = load_data( fp, _ofst, _size  );\
        _attr.vtl = FALSE;\
        _attr.is_const = TRUE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, data );\
        free( data );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

// Set generic tensor dims out of this macro.
#define NEW_NORM_TENSOR(_id, _attr, _dtype) do {\
        _attr.vtl = FALSE;\
        _attr.is_const = FALSE;\
        _attr.dtype.vx_type = _dtype;\
        _id = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO,\
                & _attr, NULL );\
        if( VSI_NN_TENSOR_ID_NA == _id ) {\
            goto error;\
        }\
    } while(0)

#define NET_NODE_NUM            (8)
#define NET_NORM_TENSOR_NUM     (2)
#define NET_CONST_TENSOR_NUM    (8)
#define NET_VIRTUAL_TENSOR_NUM  (8)
#define NET_TOTAL_TENSOR_NUM    (NET_NORM_TENSOR_NUM + NET_CONST_TENSOR_NUM + NET_VIRTUAL_TENSOR_NUM + 32)

/*-------------------------------------------
               Local Variables
 -------------------------------------------*/

/*-------------------------------------------
                  Functions
 -------------------------------------------*/
void vnn_ReleaseLenet
    (
    vsi_nn_graph_t * graph,
    vsi_bool release_ctx
    );
static uint8_t* load_data
    (
    FILE  * fp,
    size_t  ofst,
    size_t  sz
    )
{
    uint8_t* data;
    int32_t ret;
    data = NULL;
    if( NULL == fp )
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        VSILOGE("blob seek failure.");
        return NULL;
    }

    data = (uint8_t*)malloc(sz);
    if (data == NULL)
    {
        VSILOGE("buffer malloc failure.");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    return data;
} /* load_data() */

vsi_nn_graph_t * vnn_CreateLenet
    (
    const char * data_file_name,
    vsi_nn_context_t in_ctx
    )
{
    vsi_status              status;
    vsi_bool                release_ctx;
    vsi_nn_context_t        ctx;
    vsi_nn_graph_t *        graph;
    vsi_nn_node_t *         node[NET_NODE_NUM];
    vsi_nn_tensor_id_t      norm_tensor[NET_NORM_TENSOR_NUM];
    vsi_nn_tensor_id_t      const_tensor[NET_CONST_TENSOR_NUM];
    vsi_nn_tensor_attr_t    attr;
    FILE *                  fp;
    uint8_t *               data;




    ctx = NULL;
    graph = NULL;
    status = VSI_FAILURE;
    memset( &attr, 0, sizeof( attr ) );

    fp = fopen( data_file_name, "rb" );
    if( NULL == fp )
    {
        VSILOGE( "Open file %s failed.", data_file_name );
        goto error;
    }

    if( NULL == in_ctx )
    {
        ctx = vsi_nn_CreateContext();
    }
    else
    {
        ctx = in_ctx;
    }

    graph = vsi_nn_CreateGraph( ctx, NET_TOTAL_TENSOR_NUM, NET_NODE_NUM );
    if( NULL == graph )
    {
        VSILOGE( "Create graph fail." );
        goto error;
    }
    //vsi_nn_SetGraphVersion( graph, VNN_VERSION_MAJOR, VNN_VERSION_MINOR, VNN_VERSION_PATCH );
    vsi_nn_SetGraphInputs( graph, NULL, 1 );
    vsi_nn_SetGraphOutputs( graph, NULL, 1 );

/*-----------------------------------------
  Register client ops
 -----------------------------------------*/


/*-----------------------------------------
  Node definitions
 -----------------------------------------*/

    /*-----------------------------------------
      lid       - conv1_1
      var       - node[0]
      name      - conv1
      operation - convolution
      in_shape  - [[28, 28, 1, 1]]
      out_shape - [[24, 24, 20, 1]]
    -----------------------------------------*/
    NEW_VXNODE(node[0], VSI_NN_OP_CONV2D, 3, 1, 1);
    node[0]->nn_param.conv2d.ksize[0] = 5;
    node[0]->nn_param.conv2d.ksize[1] = 5;
    node[0]->nn_param.conv2d.weights = 20;
    node[0]->nn_param.conv2d.stride[0] = 1;
    node[0]->nn_param.conv2d.stride[1] = 1;
    node[0]->nn_param.conv2d.pad[0] = 0;
    node[0]->nn_param.conv2d.pad[1] = 0;
    node[0]->nn_param.conv2d.pad[2] = 0;
    node[0]->nn_param.conv2d.pad[3] = 0;
    node[0]->nn_param.conv2d.group = 1;
    node[0]->nn_param.conv2d.dilation[0] = 1;
    node[0]->nn_param.conv2d.dilation[1] = 1;
    node[0]->nn_param.conv2d.multiplier = 0;

    /*-----------------------------------------
      lid       - pool1_2
      var       - node[1]
      name      - pool1
      operation - pooling
      in_shape  - [[24, 24, 20, 1]]
      out_shape - [[12, 12, 20, 1]]
    -----------------------------------------*/
    NEW_VXNODE(node[1], VSI_NN_OP_POOL, 1, 1, 2);
    node[1]->nn_param.pool.ksize[0] = 2;
    node[1]->nn_param.pool.ksize[1] = 2;
    node[1]->nn_param.pool.stride[0] = 2;
    node[1]->nn_param.pool.stride[1] = 2;
    node[1]->nn_param.pool.pad[0] = 0;
    node[1]->nn_param.pool.pad[1] = 0;
    node[1]->nn_param.pool.pad[2] = 0;
    node[1]->nn_param.pool.pad[3] = 0;
    node[1]->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
    node[1]->nn_param.pool.round_type = VSI_NN_ROUND_CEIL;
    node[1]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING;

    /*-----------------------------------------
      lid       - conv2_3
      var       - node[2]
      name      - conv2
      operation - convolution
      in_shape  - [[12, 12, 20, 1]]
      out_shape - [[8, 8, 50, 1]]
    -----------------------------------------*/
    NEW_VXNODE(node[2], VSI_NN_OP_CONV2D, 3, 1, 3);
    node[2]->nn_param.conv2d.ksize[0] = 5;
    node[2]->nn_param.conv2d.ksize[1] = 5;
    node[2]->nn_param.conv2d.weights = 50;
    node[2]->nn_param.conv2d.stride[0] = 1;
    node[2]->nn_param.conv2d.stride[1] = 1;
    node[2]->nn_param.conv2d.pad[0] = 0;
    node[2]->nn_param.conv2d.pad[1] = 0;
    node[2]->nn_param.conv2d.pad[2] = 0;
    node[2]->nn_param.conv2d.pad[3] = 0;
    node[2]->nn_param.conv2d.group = 1;
    node[2]->nn_param.conv2d.dilation[0] = 1;
    node[2]->nn_param.conv2d.dilation[1] = 1;
    node[2]->nn_param.conv2d.multiplier = 0;

    /*-----------------------------------------
      lid       - pool2_4
      var       - node[3]
      name      - pool2
      operation - pooling
      in_shape  - [[8, 8, 50, 1]]
      out_shape - [[4, 4, 50, 1]]
    -----------------------------------------*/
    NEW_VXNODE(node[3], VSI_NN_OP_POOL, 1, 1, 4);
    node[3]->nn_param.pool.ksize[0] = 2;
    node[3]->nn_param.pool.ksize[1] = 2;
    node[3]->nn_param.pool.stride[0] = 2;
    node[3]->nn_param.pool.stride[1] = 2;
    node[3]->nn_param.pool.pad[0] = 0;
    node[3]->nn_param.pool.pad[1] = 0;
    node[3]->nn_param.pool.pad[2] = 0;
    node[3]->nn_param.pool.pad[3] = 0;
    node[3]->nn_param.pool.type = VX_CONVOLUTIONAL_NETWORK_POOLING_MAX;
    node[3]->nn_param.pool.round_type = VSI_NN_ROUND_CEIL;
    node[3]->vx_param.down_scale_size_rounding = VX_CONVOLUTIONAL_NETWORK_DS_SIZE_ROUNDING_CEILING;

    /*-----------------------------------------
      lid       - ip1_5
      var       - node[4]
      name      - ip1
      operation - fullconnect
      in_shape  - [[4, 4, 50, 1]]
      out_shape - [[500, 1]]
    -----------------------------------------*/
    NEW_VXNODE(node[4], VSI_NN_OP_FCL, 3, 1, 5);
    node[4]->nn_param.fcl.weights = 500;

    /*-----------------------------------------
      lid       - relu1_6
      var       - node[5]
      name      - relu1
      operation - relu
      in_shape  - [[500, 1]]
      out_shape - [[500, 1]]
    -----------------------------------------*/
    NEW_VXNODE(node[5], VSI_NN_OP_RELU, 1, 1, 6);

    /*-----------------------------------------
      lid       - ip2_7
      var       - node[6]
      name      - ip2
      operation - fullconnect
      in_shape  - [[500, 1]]
      out_shape - [[10, 1]]
    -----------------------------------------*/
    NEW_VXNODE(node[6], VSI_NN_OP_FCL, 3, 1, 7);
    node[6]->nn_param.fcl.weights = 10;

    /*-----------------------------------------
      lid       - prob_8
      var       - node[7]
      name      - prob
      operation - softmax
      in_shape  - [[10, 1]]
      out_shape - [[10, 1]]
    -----------------------------------------*/
    NEW_VXNODE(node[7], VSI_NN_OP_SOFTMAX, 1, 1, 8);
    node[7]->nn_param.softmax.beta = 1.0;


/*-----------------------------------------
  Tensor initialize
 -----------------------------------------*/
    attr.dtype.fmt = VSI_NN_DIM_FMT_NCHW;
    /* @input_0:out0 */
    attr.size[0] = 28;
    attr.size[1] = 28;
    attr.size[2] = 1;
    attr.size[3] = 1;
    attr.dim_num = 4;
    attr.dtype.scale = 0.00390625;
    attr.dtype.zero_point = 0;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_NORM_TENSOR(norm_tensor[0], attr, VSI_NN_TYPE_UINT8);

    /* @output_9:out0 */
    attr.size[0] = 10;
    attr.size[1] = 1;
    attr.dim_num = 2;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
    NEW_NORM_TENSOR(norm_tensor[1], attr, VSI_NN_TYPE_FLOAT16);



    /* @conv1_1:weight */
    attr.size[0] = 5;
    attr.size[1] = 5;
    attr.size[2] = 1;
    attr.size[3] = 20;
    attr.dim_num = 4;
    attr.dtype.scale = 0.003362337;
    attr.dtype.zero_point = 119;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_CONST_TENSOR(const_tensor[0], attr, VSI_NN_TYPE_UINT8, 80, 500);

    /* @conv1_1:bias */
    attr.size[0] = 20;
    attr.dim_num = 1;
    attr.dtype.scale = 1.3134e-05;
    attr.dtype.zero_point = 0;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_CONST_TENSOR(const_tensor[1], attr, VSI_NN_TYPE_INT32, 0, 80);

    /* @conv2_3:weight */
    attr.size[0] = 5;
    attr.size[1] = 5;
    attr.size[2] = 20;
    attr.size[3] = 50;
    attr.dim_num = 4;
    attr.dtype.scale = 0.001148205;
    attr.dtype.zero_point = 128;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_CONST_TENSOR(const_tensor[2], attr, VSI_NN_TYPE_UINT8, 780, 25000);

    /* @conv2_3:bias */
    attr.size[0] = 50;
    attr.dim_num = 1;
    attr.dtype.scale = 2.2138e-05;
    attr.dtype.zero_point = 0;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_CONST_TENSOR(const_tensor[3], attr, VSI_NN_TYPE_INT32, 580, 200);

    /* @ip1_5:weight */
    attr.size[0] = 800;
    attr.size[1] = 500;
    attr.dim_num = 2;
    attr.dtype.scale = 0.000735485;
    attr.dtype.zero_point = 130;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_CONST_TENSOR(const_tensor[4], attr, VSI_NN_TYPE_UINT8, 27780, 400000);

    /* @ip1_5:bias */
    attr.size[0] = 500;
    attr.dim_num = 1;
    attr.dtype.scale = 2.9977e-05;
    attr.dtype.zero_point = 0;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_CONST_TENSOR(const_tensor[5], attr, VSI_NN_TYPE_INT32, 25780, 2000);

    /* @ip2_7:weight */
    attr.size[0] = 500;
    attr.size[1] = 10;
    attr.dim_num = 2;
    attr.dtype.scale = 0.001580426;
    attr.dtype.zero_point = 135;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_CONST_TENSOR(const_tensor[6], attr, VSI_NN_TYPE_UINT8, 427820, 5000);

    /* @ip2_7:bias */
    attr.size[0] = 10;
    attr.dim_num = 1;
    attr.dtype.scale = 3.1484e-05;
    attr.dtype.zero_point = 0;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_CONST_TENSOR(const_tensor[7], attr, VSI_NN_TYPE_INT32, 427780, 40);



    /* @conv1_1:out0 */
    attr.dtype.scale = 0.019280685;
    attr.dtype.zero_point = 140;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_VIRTUAL_TENSOR(node[0]->output.tensors[0], attr, VSI_NN_TYPE_UINT8);

    /* @pool1_2:out0 */
    attr.dtype.scale = 0.019280685;
    attr.dtype.zero_point = 140;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_VIRTUAL_TENSOR(node[1]->output.tensors[0], attr, VSI_NN_TYPE_UINT8);

    /* @conv2_3:out0 */
    attr.dtype.scale = 0.040758733;
    attr.dtype.zero_point = 141;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_VIRTUAL_TENSOR(node[2]->output.tensors[0], attr, VSI_NN_TYPE_UINT8);

    /* @pool2_4:out0 */
    attr.dtype.scale = 0.040758733;
    attr.dtype.zero_point = 141;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_VIRTUAL_TENSOR(node[3]->output.tensors[0], attr, VSI_NN_TYPE_UINT8);

    /* @ip1_5:out0 */
    attr.dtype.scale = 0.019920889;
    attr.dtype.zero_point = 0;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_VIRTUAL_TENSOR(node[4]->output.tensors[0], attr, VSI_NN_TYPE_UINT8);

    /* @relu1_6:out0 */
    attr.dtype.scale = 0.019920889;
    attr.dtype.zero_point = 0;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_VIRTUAL_TENSOR(node[5]->output.tensors[0], attr, VSI_NN_TYPE_UINT8);

    /* @ip2_7:out0 */
    attr.dtype.scale = 0.062514879;
    attr.dtype.zero_point = 80;
    attr.dtype.qnt_type = VSI_NN_QNT_TYPE_AFFINE_ASYMMETRIC;
    NEW_VIRTUAL_TENSOR(node[6]->output.tensors[0], attr, VSI_NN_TYPE_UINT8);



/*-----------------------------------------
  Connection initialize
 -----------------------------------------*/
    node[0]->input.tensors[0] = norm_tensor[0];
    node[7]->output.tensors[0] = norm_tensor[1];

    /* conv1_1 */
    node[0]->input.tensors[1] = const_tensor[0]; /* data_weight */
    node[0]->input.tensors[2] = const_tensor[1]; /* data_bias */

    /* pool1_2 */
    node[1]->input.tensors[0] = node[0]->output.tensors[0];

    /* conv2_3 */
    node[2]->input.tensors[0] = node[1]->output.tensors[0];
    node[2]->input.tensors[1] = const_tensor[2]; /* data_weight */
    node[2]->input.tensors[2] = const_tensor[3]; /* data_bias */

    /* pool2_4 */
    node[3]->input.tensors[0] = node[2]->output.tensors[0];

    /* ip1_5 */
    node[4]->input.tensors[0] = node[3]->output.tensors[0];
    node[4]->input.tensors[1] = const_tensor[4]; /* data_weight */
    node[4]->input.tensors[2] = const_tensor[5]; /* data_bias */

    /* relu1_6 */
    node[5]->input.tensors[0] = node[4]->output.tensors[0];

    /* ip2_7 */
    node[6]->input.tensors[0] = node[5]->output.tensors[0];
    node[6]->input.tensors[1] = const_tensor[6]; /* data_weight */
    node[6]->input.tensors[2] = const_tensor[7]; /* data_bias */

    /* prob_8 */
    node[7]->input.tensors[0] = node[6]->output.tensors[0];



    graph->input.tensors[0] = norm_tensor[0];
    graph->output.tensors[0] = norm_tensor[1];


    status = vsi_nn_SetupGraph( graph, FALSE );
    if( VSI_FAILURE == status )
    {
        goto error;
    }

    fclose( fp );

    return graph;

error:
    if( NULL != fp )
    {
        fclose( fp );
    }

    release_ctx = ( NULL == in_ctx );
    vsi_nn_DumpGraphToJson(graph);
    vnn_ReleaseLenet( graph, release_ctx );

    return NULL;
} /* vsi_nn_CreateLenetAsymmetricQuantizedU8() */

void vnn_ReleaseLenet
    (
    vsi_nn_graph_t * graph,
    vsi_bool release_ctx
    )
{
    vsi_nn_context_t ctx;
    if( NULL != graph )
    {
        ctx = graph->ctx;
        vsi_nn_ReleaseGraph( &graph );

        /*-----------------------------------------
        Unregister client ops
        -----------------------------------------*/
        if( release_ctx )
        {
            vsi_nn_ReleaseContext( &ctx );
        }
    }
} /* vsi_nn_ReleaseLenetAsymmetricQuantizedU8() */


using shared_context = std::shared_ptr< std::pointer_traits<vsi_nn_context_t>::element_type >;

thread_local shared_context thread_local_context;

struct deleter {
    void operator()(vsi_nn_context_t ctx)
    {
        VSILOGD("Release context.");
        vsi_nn_ReleaseContext(&ctx);
    }
};

void run_graph(vsi_nn_graph_t* g, shared_context ctx) {
    VSILOGD("Run thread start ... ");
    int e = vsi_nn_VerifyGraph(g);
    VSILOGD("Verify graph error code %#x", e);
    e = vsi_nn_RunGraph(g);
    VSILOGD("Run graph error code %#x", e);
    VSILOGD("Release graph");
    vnn_ReleaseLenet(g, false);
    VSILOGD("Run graph complete");
}

void test(const char* data_file, shared_context ctx) {
    vsi_nn_graph_t* graph = vnn_CreateLenet(
            data_file, ctx.get());
    std::thread(run_graph, graph, ctx).detach();
    VSILOGD("Test thread finish");
}

int main(int argc, char** argv)
{
    assert(argc == 2);
    thread_local_context.reset(vsi_nn_CreateContext(), deleter());
    const int threads_of_processing_graph = 2;
    for (int i = 0; i < threads_of_processing_graph; i++){
        std::thread(test, argv[1], thread_local_context).detach();
        VSILOGD("Ctx use_count: %d", thread_local_context.use_count());
    }
    sleep(5);
    VSILOGD("Main return ctx use_count: %d", thread_local_context.use_count());
    return 0;
}
