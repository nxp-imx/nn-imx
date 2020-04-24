kernel_meta = '''\
#define _INPUT_NUM          (1)
#define _OUTPUT_NUM         (1)
#define _KERNEL_NAME        CVIVANTE_NAMESPACE("%lower(KERNEL_TYPE)%.%KERNEL_NAME%")
'''

kernel_initializer = ''

kernel_query = '''\
static vsi_status _query_kernel
    (
    vsi_nn_kernel_t * kernel,
    vsi_nn_tensor_t * const * const inputs,
    vsi_nn_tensor_t * const * const outputs
    /* Add extra params */
    )
{
    vsi_status status = VSI_FAILURE;
    snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  _KERNEL_NAME );
    kernel->info.function    = _compute;
    kernel->info.parameters  = _%KERNEL_NAME%_kernel_param_def;
    kernel->info.numParams   = _cnt_of_array( _%KERNEL_NAME%_kernel_param_def );
    //status = VSI_SUCCESS;
    return status;
} /* _query_kernel() */
'''

kernel_check = ''

kernel_function = '''\

/*
 * Kernel function
 */
DEF_KERNEL_EXECUTOR(_compute)
    (
    vsi_nn_kernel_node_t                node,
    const vsi_nn_kernel_node_param_t  * param,
    size_t                              param_size
    )
{
    vsi_status status = VSI_FAILURE;
    // vsi_nn_kernel_tensor_attr * attr[2] = { NULL };
    // attr[0] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[0] );
    // attr[1] = vsi_nn_kernel_tensor_attr_create( (vsi_nn_kernel_tensor_t)param[1] );

    // Do something

    // vsi_nn_kernel_tensor_attr_release( &attr[0] );
    // vsi_nn_kernel_tensor_attr_release( &attr[1] );
    return status;
} /* _compute() */
'''

