from template import kernel_gpu
kernel_meta = kernel_gpu.kernel_meta

kernel_initializer = kernel_gpu.kernel_initializer

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
    size_t param_def_size               = _cnt_of_array( _%KERNEL_NAME%_kernel_param_def );
    vx_kernel_initialize_f  initializer = _%KERNEL_NAME%_initializer;

    uint32_t key;
    int i;

    in_dtype  = vsi_nn_kernel_map_dtype( inputs[0]->attr.dtype.vx_type );
    out_dtype = vsi_nn_kernel_map_dtype( outputs[0]->attr.dtype.vx_type );

    key = %upper(KERNEL_NAME)%_HASH_KEY( in_dtype, out_dtype );

    for( i = 0; i < kernel_map_size; i ++ )
    {
        if( kernel_map[i].key == key )
        {
            break;
        }
    }
    if( i < kernel_map_size )
    {
        snprintf( kernel->info.name, VX_MAX_KERNEL_NAME, "%s",  kernel_map[i].function_name );
        kernel->info.parameters  = param_def;
        kernel->info.numParams   = param_def_size;
        kernel->info.initialize  = initializer;
        // Register code source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_CODE, 2,
                "vsi_nn_kernel_header",
                kernel_map[i].source_name );
        // Register binary source
        vsi_nn_kernel_add_source( kernel, VSI_NN_GPU_SOURCE_FMT_EXECUTABLE, 1,
                kernel_map[i].source_name );
        status = VSI_SUCCESS;
    }
    return status;
} /* _query_kernel() */
'''

kernel_check = kernel_gpu.kernel_check

kernel_function = kernel_gpu.kernel_function

