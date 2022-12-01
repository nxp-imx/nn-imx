__kernel void gemm_transa_F32F32toF32_2D(
    __read_only image2d_t   inputA,
    __read_only image2d_t   inputB,
    __write_only image2d_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);
    float4 sum = (float4)(0);

    for(; coord.z < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord.yz);
        tempB0 = read_imagef(inputB, coord.xz);
        coord.z++;

        sum = sum + tempA0 * tempB0;
    }
    write_imagef(output, coord.xy, sum);
}

__kernel void gemm_transa_F32F32toF32_3D(
    __read_only image2d_array_t   inputA,
    __read_only image2d_array_t   inputB,
    __write_only image2d_array_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int4 coord_a = (int4)(gidy, 0, (ac2zero ? 0 : get_global_id(2)), 0);
    int4 coord_b = (int4)(gidx, 0, (bc2zero ? 0 : get_global_id(2)), 0);

    float4 sum = (float4)(0);

    for(; coord_a.y < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = read_imagef(inputA, coord_a);
        tempB0 = read_imagef(inputB, coord_b);
        coord_a.y++;
        coord_b.y++;

        sum = sum + tempA0 * tempB0;
    }

    coord_b.y = gidy;
    coord_b.z = get_global_id(2);
    write_imagef(output, coord_b, sum);
}

__kernel void gemm_transa_I8I8toI8_2D(
    __read_only image2d_t   inputA,
    __read_only image2d_t   inputB,
    __write_only image2d_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int4 coord = (int4)(get_global_id(0), get_global_id(1), 0, 0);
    float4 sum = (float4)(0);
    int4 dst;

    for(; coord.z < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = convert_float4(read_imagei(inputA, coord.yz));
        tempB0 = convert_float4(read_imagei(inputB, coord.xz));
        tempA0.x = (tempA0.x - zp_a) * scale_a;
        tempB0.x = (tempB0.x - zp_b) * scale_b;
        coord.z++;

        sum = sum + tempA0 * tempB0;
    }
    sum.x = sum.x * scale_out + zp_out;
    dst = convert_int4(sum);

    write_imagei(output, coord.xy, dst);
}

__kernel void gemm_transa_I8I8toI8_3D(
    __read_only image2d_array_t   inputA,
    __read_only image2d_array_t   inputB,
    __write_only image2d_array_t  output,
    int M,
    int K,
    int N,
    int ac2zero,
    int bc2zero,
    float scale_a,
    float zp_a,
    float scale_b,
    float zp_b,
    float scale_out,
    float zp_out
    )
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int4 coord_a = (int4)(gidy, 0, (ac2zero ? 0 : get_global_id(2)), 0);
    int4 coord_b = (int4)(gidx, 0, (bc2zero ? 0 : get_global_id(2)), 0);

    float4 sum = (float4)(0);
    int4 dst;

    for(; coord_a.y < K;)
    {
        float4 tempA0;
        float4 tempB0;

        tempA0 = convert_float4(read_imagei(inputA, coord_a));
        tempB0 = convert_float4(read_imagei(inputB, coord_b));
        tempA0.x = (tempA0.x - zp_a) * scale_a;
        tempB0.x = (tempB0.x - zp_b) * scale_b;
        coord_a.y++;
        coord_b.y++;

        sum = sum + tempA0 * tempB0;
    }
    sum.x = sum.x * scale_out + zp_out;
    dst = convert_int4(sum);

    coord_b.y = gidy;
    coord_b.z = get_global_id(2);
    write_imagei(output, coord_b, dst);
}