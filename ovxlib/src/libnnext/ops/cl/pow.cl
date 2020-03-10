__kernel void pow_FP32FP32toFP32
    (
    __read_only  image2d_array_t    input0,
    __read_only  image2d_array_t    input1,
    __write_only image2d_array_t    output
    )
{
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0);

    float4 src0;
    float4 src1;
    readImage2DArray(src0, input0, coord);
    readImage2DArray(src1, input1, coord);
    
    float4 dst = exp2(src1*log2(src0));

    write_imagef(output, coord, dst);
}

__kernel void pow_FP32FP32toFP32_2D
    (
    __read_only  image2d_t    input0,
    __read_only  image2d_t    input1,
    __write_only image2d_t    output
    )
{
    int2 coord =  (int2)(get_global_id(0), get_global_id(1));

    float4 src0 = read_imagef(input0, coord);
    float4 src1 = read_imagef(input1, coord);

    float4 dst = exp2(src1*log2(src0));

    write_imagef(output, coord, dst);
}
