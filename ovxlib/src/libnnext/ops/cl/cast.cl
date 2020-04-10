
#define CAST_FUN(src_name, dst_name, src_type, dst_type, conv_type, read_fun, write_fun) \
__kernel void cast_##src_name##to##dst_name( \
    __read_only  image2d_array_t  input, \
    __write_only image2d_array_t  output) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    src_type src = read_fun(input, coord); \
    dst_type dst = 0; \
    int4  tmp = convert_int4(src);\
    dst.x = (conv_type)(tmp.x); \
    write_fun(output, coord, dst); \
}

CAST_FUN(F32, I32, float4, int4,   int,   read_imagef,  write_imagei)
CAST_FUN(F32, U32, float4, uint4,  unsigned int,  read_imagef,  write_imageui)
CAST_FUN(F32, I16, float4, int4,   short,   read_imagef,  write_imagei)
CAST_FUN(F32, U16, float4, uint4,  unsigned short,  read_imagef,  write_imageui)
CAST_FUN(F32, I8,  float4, int4,   char,   read_imagef,  write_imagei)
CAST_FUN(F32, U8,  float4, uint4,  unsigned char,  read_imagef,  write_imageui)
CAST_FUN(I32, I32, int4,   int4,   int,   read_imagei,  write_imagei)
CAST_FUN(I32, U32, int4,   uint4,  unsigned int,  read_imagei,  write_imageui)
CAST_FUN(I32, I16, int4,   int4,   short,   read_imagei,  write_imagei)
CAST_FUN(I32, U16, int4,   uint4,  unsigned short,  read_imagei,  write_imageui)
CAST_FUN(I32, I8,  int4,   int4,   char,   read_imagei,  write_imagei)
CAST_FUN(I32, U8,  int4,   uint4,  unsigned char,  read_imagei,  write_imageui)
CAST_FUN(U32, I32, uint4,  int4,   int,   read_imageui, write_imagei)
CAST_FUN(U32, U32, uint4,  uint4,  unsigned int,  read_imageui, write_imageui)
CAST_FUN(U32, I16, uint4,  int4,   short,   read_imageui, write_imagei)
CAST_FUN(U32, U16, uint4,  uint4,  unsigned short,  read_imageui, write_imageui)
CAST_FUN(U32, I8,  uint4,  int4,   char,   read_imageui, write_imagei)
CAST_FUN(U32, U8,  uint4,  uint4,  unsigned char,  read_imageui, write_imageui)

#define CAST_ToFloat_FUN(src_name, dst_name, src_type, read_fun) \
__kernel void cast_##src_name##to##dst_name( \
    __read_only  image2d_array_t  input, \
    __write_only image2d_array_t  output) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    src_type src = read_fun(input, coord); \
    float4 dst = 0; \
    dst.x = (float)(src.x); \
    write_imagef(output, coord, dst); \
}

CAST_ToFloat_FUN(F32, F32, float4, read_imagef)
CAST_ToFloat_FUN(I32, F32, int4,   read_imagei)
CAST_ToFloat_FUN(U32, F32, uint4,  read_imageui)

#define CAST_FUN_2D(src_name, dst_name, src_type, dst_type, conv_type, read_fun, write_fun) \
__kernel void cast_##src_name##to##dst_name##_2D( \
    __read_only  image2d_t  input, \
    __write_only image2d_t  output) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
    src_type src = read_fun(input, coord); \
    dst_type dst = 0; \
    int4  tmp = convert_int4(src);\
    dst.x = (conv_type)(tmp.x); \
    write_fun(output, coord, dst); \
}

CAST_FUN_2D(F32, I32, float4, int4,   int,   read_imagef,  write_imagei)
CAST_FUN_2D(F32, U32, float4, uint4,  unsigned int,  read_imagef,  write_imageui)
CAST_FUN_2D(F32, I16, float4, int4,   short,   read_imagef,  write_imagei)
CAST_FUN_2D(F32, U16, float4, uint4,  unsigned short,  read_imagef,  write_imageui)
CAST_FUN_2D(F32, I8,  float4, int4,   char,   read_imagef,  write_imagei)
CAST_FUN_2D(F32, U8,  float4, uint4,  unsigned char,  read_imagef,  write_imageui)
CAST_FUN_2D(I32, I32, int4,   int4,   int,   read_imagei,  write_imagei)
CAST_FUN_2D(I32, U32, int4,   uint4,  unsigned int,  read_imagei,  write_imageui)
CAST_FUN_2D(I32, I16, int4,   int4,   short,   read_imagei,  write_imagei)
CAST_FUN_2D(I32, U16, int4,   uint4,  unsigned short,  read_imagei,  write_imageui)
CAST_FUN_2D(I32, I8,  int4,   int4,   char,   read_imagei,  write_imagei)
CAST_FUN_2D(I32, U8,  int4,   uint4,  unsigned char,  read_imagei,  write_imageui)
CAST_FUN_2D(U32, I32, uint4,  int4,   int,   read_imageui, write_imagei)
CAST_FUN_2D(U32, U32, uint4,  uint4,  unsigned int,  read_imageui, write_imageui)
CAST_FUN_2D(U32, I16, uint4,  int4,   short,   read_imageui, write_imagei)
CAST_FUN_2D(U32, U16, uint4,  uint4,  unsigned short,  read_imageui, write_imageui)
CAST_FUN_2D(U32, I8,  uint4,  int4,   char,   read_imageui, write_imagei)
CAST_FUN_2D(U32, U8,  uint4,  uint4,  unsigned char,  read_imageui, write_imageui)

#define CAST_ToFloat_FUN_2D(src_name, dst_name, src_type, read_fun) \
__kernel void cast_##src_name##to##dst_name##_2D( \
    __read_only  image2d_t  input, \
    __write_only image2d_t  output) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
    src_type src = read_fun(input, coord); \
    float4 dst = 0; \
    dst.x = (float)(src.x); \
    write_imagef(output, coord, dst); \
}

CAST_ToFloat_FUN_2D(F32, F32, float4, read_imagef)
CAST_ToFloat_FUN_2D(I32, F32, int4,   read_imagei)
CAST_ToFloat_FUN_2D(U32, F32, uint4,  read_imageui)

#define CAST_TO_BOOL_FUN(src_name, src_type, read_fun) \
__kernel void cast_##src_name##toBOOL8( \
    __read_only  image2d_array_t  input, \
    __write_only image2d_array_t  output) \
{ \
    int4 coord =  (int4)(get_global_id(0), get_global_id(1), get_global_id(2), 0); \
    src_type src = read_fun(input, coord); \
    int4 dst = 0; \
    dst.x = (int)(src.x != 0); \
    write_imagei(output, coord, dst); \
}

CAST_TO_BOOL_FUN(F32, float4, read_imagef)
CAST_TO_BOOL_FUN(I32, int4,   read_imagei)
CAST_TO_BOOL_FUN(U32, uint4,  read_imageui)


#define CAST_TO_BOOL_FUN_2D(src_name, src_type, read_fun) \
__kernel void cast_##src_name##toBOOL8_2D( \
    __read_only  image2d_t  input, \
    __write_only image2d_t  output) \
{ \
    int2 coord =  (int2)(get_global_id(0), get_global_id(1)); \
    src_type src = read_fun(input, coord); \
    int4 dst = 0; \
    dst.x = (int)(src.x != 0); \
    write_imagei(output, coord, dst); \
}

CAST_TO_BOOL_FUN_2D(F32, float4, read_imagef)
CAST_TO_BOOL_FUN_2D(I32, int4,   read_imagei)
CAST_TO_BOOL_FUN_2D(U32, uint4,  read_imageui)

