
__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void l2normalizescale_axis0_F32_F32toF32_2D(
    __read_only  image2d_t input,
    __read_only  image2d_t scale,
    __write_only image2d_t output,
                       int axis,
                       int axis_size,
                     float rsEps
    )
{
    int lidx = get_local_id(0);
    int gidx = get_global_id(0);
    float4 src, scale_value, result;
    float sum  = 0.0f, pSum = 0.0f, rsqrt_sum = 0.0f;
    int2 coord = (int2)(gidx, get_global_id(1));
    int2 coord_scale = (int2)(gidx, 0);
    __local float lcl_sum[16];
    for(; coord.x < axis_size; coord.x += 16)
    {
        src = read_imagef(input, coord);
        pSum += (src.x * src.x);
    }
    lcl_sum[lidx] = pSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 *pLocalPtr = (float4 *)&lcl_sum[0];
    float4 one = (float4)(1, 1, 1, 1);
    float4 data0;
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];
    sum = dot(data0, one);
    rsqrt_sum = (sum == 0 ? rsEps : rsqrt(sum));
    for(coord.x = gidx; coord.x < axis_size; coord.x += 16)
    {
        src         = read_imagef(input, coord);
        scale_value = read_imagef(scale, coord_scale);
        result      = src * rsqrt_sum * scale_value;
        write_imagef(output, coord, result.xxxx);
    }
}

__kernel __attribute__((reqd_work_group_size(16, 1, 1))) void l2normalizescale_axis0_U8_F32toU8_2D(
    __read_only  image2d_t input,
    __read_only  image2d_t scale,
    __write_only image2d_t output,
                       int axis,
                       int axis_size,
                     float rsEps,
                     float inputScale,
                     float inputTail,
                     float outputScale,
                     float outputZP
    )
{
    int lidx = get_local_id(0);
    int gidx = get_global_id(0);
    float4 src, scale_value, result;
    float sum  = 0.0f, pSum = 0.0f, rsqrt_sum = 0.0f;
    int2 coord = (int2)(gidx, get_global_id(1));
    int2 coord_scale = (int2)(gidx, 0);
    __local float lcl_sum[16];
    for(; coord.x < axis_size; coord.x += 16)
    {
        src = convert_float4(read_imageui(input, coord))  * inputScale + inputTail;
        pSum += (src.x * src.x);
    }
    lcl_sum[lidx] = pSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    float4 *pLocalPtr = (float4 *)&lcl_sum[0];
    float4 one = (float4)(1, 1, 1, 1);
    float4 data0;
    data0 = pLocalPtr[0] + pLocalPtr[1] + pLocalPtr[2] + pLocalPtr[3];
    sum = dot(data0, one);
    rsqrt_sum = (sum == 0 ? rsEps : rsqrt(sum));
    for(coord.x = gidx; coord.x < axis_size; coord.x += 16)
    {
        src         = convert_float4(read_imageui(input, coord))  * inputScale + inputTail;
        scale_value = read_imagef(scale, coord_scale);
        result      = src * rsqrt_sum * scale_value;
        uint4 dst = convert_uint4_rte(result * outputScale + outputZP);
        write_imageui(output, coord, dst);
    }
}


