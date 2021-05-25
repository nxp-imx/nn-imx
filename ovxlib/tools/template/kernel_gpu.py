kernel_meta = '''\
typedef enum
{
    INTERNAL_KERNEL_%upper(KERNEL_NAME)%,
} _internal_kernel_e;

#define _%upper(KERNEL_NAME)%_KERNEL_SOURCE      "%KERNEL_NAME%"
#define _%upper(KERNEL_NAME)%_KERNEL_NAME        CVIVANTE_NAMESPACE("%lower(KERNEL_TYPE)%.%KERNEL_NAME%")

// Add kernel hashtable here
#define %upper(KERNEL_NAME)%_HASH_KEY( IN_DTYPE, OUT_DTYPE ) \\
        (( IN_DTYPE << 8 ) | ( OUT_DTYPE ))
#define PACK_KERNEL_MAP( IN_DTYPE, OUT_DTYPE, SOURCE ) \\
        { %upper(KERNEL_NAME)%_HASH_KEY( IN_DTYPE, OUT_DTYPE ), _%upper(KERNEL_NAME)%_KERNEL_NAME, SOURCE }

typedef struct
{
    uint32_t key;
    char * function_name;
    const char * source_name;
} _kernel_map_type;

static const _kernel_map_type _%KERNEL_NAME%_kernel_map[] =
{
    // Register kernel here
    PACK_KERNEL_MAP( F32, F32, _%upper(KERNEL_NAME)%_KERNEL_SOURCE ),
};
'''

kernel_initializer = '''\

/*
 * Kernel initializer
 */
DEF_KERNEL_INITIALIZER(_%KERNEL_NAME%_initializer)
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

    // Add initializer

    // vsi_nn_kernel_tensor_attr_release( &attr[0] );
    // vsi_nn_kernel_tensor_attr_release( &attr[1] );
    return status;
} /* _%KERNEL_NAME%_initializer() */
'''

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
    vsi_nn_kernel_dtype_e in_dtype;
    vsi_nn_kernel_dtype_e out_dtype;
    const _kernel_map_type * kernel_map = _%KERNEL_NAME%_kernel_map;
    size_t kernel_map_size              = _cnt_of_array( _%KERNEL_NAME%_kernel_map );
    vx_param_description_t * param_def  = _%KERNEL_NAME%_kernel_param_def;
    vx_kernel_initialize_f  initializer = _%KERNEL_NAME%_initializer;

    uint32_t key;
    uint32_t i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = %upper(KERNEL_NAME)%_HASH_KEY( in_dtype, out_dtype );

    for ( i = 0; i < (uint32_t)kernel_map_size; i ++ )
    {
        if ( kernel_map[i].key == key )
        {
            break;
        }
    }
    if ( i < (uint32_t)kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = _cnt_of_array( _%KERNEL_NAME%_kernel_param_def );
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 1,
                kernel_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */
'''

kernel_check = '''\
    /*
    // Check if gpu can support the size
    if( !vsi_nn_kernel_gpu_check_shape(
        (int32_t*)inputs[0]->attr.size, inputs[0]->attr.dim_num ) )
    {
        return NULL;
    }
    */
'''

kernel_function = ''

