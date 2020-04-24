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
#include "jpeglib.h"

#include "vsi_nn_pub.h"

#include "vnn_global.h"
#include "vnn_pre_process.h"

#include <VX/vx_khr_cnn.h>
#define _BASETSD_H
/*-------------------------------------------
                  Functions
-------------------------------------------*/

static vx_enum vnn_GetFileType(vx_char *file_name)
{
vx_enum type = 0;
vx_char *ptr;
vx_char suffix = '.';
vx_uint32 pos,n;

vx_char buff[32] = {0};

ptr = strrchr(file_name, suffix);
pos = ptr - file_name;

n = strlen(file_name) - (pos + 1);
strncpy(buff, file_name+(pos+1), n);

if(strcmp(buff, "jpg") == 0 || strcmp(buff, "jpeg") == 0)
    type = NN_FILE_JPG;
else if(strcmp(buff, "tensor") == 0)
    type = NN_FILE_TENSOR;
else
    type = NN_FILE_NONE;

return type;
}

static int vnn_ConvertJpegToBmpData
    (
    FILE * inputFile,
    unsigned char* bmpData,
    uint32_t *bmpWidth, 
    uint32_t *bmpHeight
    )
{
struct jpeg_decompress_struct cinfo;
struct jpeg_error_mgr jerr;
JSAMPARRAY buffer;
unsigned char *point = NULL;
unsigned long width, height;
unsigned short depth = 0;

cinfo.err = jpeg_std_error(&jerr);
jpeg_create_decompress(&cinfo);
jpeg_stdio_src(&cinfo,inputFile);
jpeg_read_header(&cinfo,TRUE);

if (bmpData == NULL)
    {
    width = cinfo.image_width;
    height = cinfo.image_height;
    }
else
    {
    jpeg_start_decompress(&cinfo);

    width  = cinfo.output_width;
    height = cinfo.output_height;
    depth  = cinfo.output_components;


    buffer = (*cinfo.mem->alloc_sarray)
        ((j_common_ptr)&cinfo, JPOOL_IMAGE, width*depth, 1);

    point = bmpData;

    while (cinfo.output_scanline < height)
        {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(point, *buffer, width * depth);
        point += width * depth;
        }

    jpeg_finish_decompress(&cinfo);
    }

jpeg_destroy_decompress(&cinfo);

if (bmpWidth != NULL) *bmpWidth = width;
if (bmpHeight != NULL) *bmpHeight = height;
return depth;
}

static vx_dtype *vnn_ImagedataToDtype
    (
    vsi_nn_tensor_t *tensor,
    vx_uint8 *data,
    vx_float32 *mean_value,
    vx_uint32 width, 
    vx_uint32 height, 
    vx_uint32 channel
    )
{
vx_status status = VX_FAILURE;
vx_uint32   i, j, sz, offset;
vx_dtype *Dbuffer = NULL;

sz = width * height * channel;
Dbuffer = (vx_dtype *)malloc(sz * sizeof(vx_dtype));
_CHECK_PTR(Dbuffer, error);
memset(Dbuffer, 0, sz * sizeof(vx_dtype));

if(mean_value)
    {
    vx_float32 mean,scale,val;
    scale = mean_value[3];

    for(i = 0; i < channel; i ++)
        {
        mean = mean_value[i];
        offset = width * height * i;
        for(j = 0; j < width * height; j ++)
            {
            val = ((vx_float32)data[offset + j] - mean) / scale;
            status = vsi_nn_Fp32toDtype(val, Dbuffer, (offset + j), &tensor->attr.dtype);
            _CHECK_STATUS(status, error);
            }
        }
    }
else
    {
    for(i = 0; i < sz; i ++)
        {
        status = vsi_nn_Fp32toDtype((vx_float32)data[i], Dbuffer, i, &tensor->attr.dtype);
        _CHECK_STATUS(status, error);
        }
    }

return Dbuffer;
error:
if(Dbuffer)free(Dbuffer);
return NULL;
}

/*
    Transpose the RGB888 data for driver
    RGBRGBRGB --> reorder[2 1 0]: BBBGGGRRR --- caffe
    RGBRGBRGB --> reorder[0 1 2]: RRRGGGBBB --- tensorflow
*/
static void vnn_ImagedataTranspose
    (
    vx_uint8 *bmp_data,
    vx_uint32 *reorder,
    vx_uint32 width, 
    vx_uint32 height, 
    vx_uint32 channels
    )
{
vx_uint32   i, j, offset, sz, order;
vx_uint8 *data;

sz = width * height * channels;
data = (vx_uint8 *)malloc(sz * sizeof(vx_uint8));
if(data == NULL) return ;
memset(data, 0, sizeof(vx_uint8) * sz);

for(i = 0; i < channels; i ++)
    {
    if(reorder)
        order = reorder[i];
    else
        order = i;

    offset = width * height * i;

    for(j = 0; j < width * height; j ++)
        {
        data[j + offset] = bmp_data[j * channels + order];
        }
    }

memcpy(bmp_data, data, sz * sizeof(vx_uint8));
if(data)free(data);
}

/*
    jpg file --> BMP data(dataformat: RGBRGBRGB...)
*/
static vx_uint8 *vnn_ReadJpegImage
    (
    vx_char *name, 
    vx_uint32 width, 
    vx_uint32 height, 
    vx_uint32 channels
    )
{
vx_uint8   *bmpData;
vx_uint32   sz;
FILE *bmpFile;

bmpData = NULL;
bmpFile = NULL;
sz = width * height * channels;

bmpFile = fopen( name, "rb" );
_CHECK_PTR(bmpFile, final);

bmpData = (vx_uint8 *)malloc( sz * sizeof( vx_uint8 ) );
_CHECK_PTR(bmpData, final);
memset(bmpData, 0, sizeof( vx_uint8 ) * sz);

vnn_ConvertJpegToBmpData( bmpFile, bmpData, NULL, NULL );

final:
if(bmpFile)fclose(bmpFile);
return bmpData;
}

static vx_dtype *vnn_ReadTensorImage
    (
    vsi_nn_tensor_t *tensor,
    vx_char *name,
    vx_uint32 *size,
    vx_uint32 dim
    )
{
vx_status status = VX_FAILURE;
vx_uint32 i = 0;
vx_float32 fval = 0.0;
vx_dtype *tensorData;
vx_uint32 sz = 1;
FILE *tensorFile;

tensorData = NULL;
tensorFile = fopen(name, "rb");
_CHECK_PTR(tensorFile, error);

for(i = 0; i < dim; i++)
    sz *= size[i];

tensorData = (vx_dtype *)malloc(sz * sizeof(vx_dtype));
_CHECK_PTR(tensorData, error);
memset(tensorData, 0, sz * sizeof(vx_dtype));

for(i = 0; i < sz; i++)
    {
    status = fscanf( tensorFile, "%f ", &fval );
    status = vsi_nn_Fp32toDtype(fval, tensorData, i, &tensor->attr.dtype);
    _CHECK_STATUS(status, error);
    }

if(tensorFile)fclose(tensorFile);
return tensorData;
error:
if(tensorFile)fclose(tensorFile);
return NULL;
}

vx_status vnn_PrePocessLenet
    (
    vsi_nn_graph_t *graph,
    vx_char *image_name
    )
{
vx_status status = VX_FAILURE;
vx_enum fileType;
vsi_nn_tensor_t *tensor;
vx_dtype *data = NULL;
vx_uint8 *bmpData = NULL;
vx_float32 mean_value[4] = {0.0,0.0,0.0,256.0};
vx_uint32 reorder[3] = {0,1,2};

tensor = vsi_nn_GetTensor( graph, graph->input.tensors[0] );
fileType = vnn_GetFileType(image_name);
if(fileType == NN_FILE_JPG)
    {
    vx_uint32 width     = tensor->attr.size[0];
    vx_uint32 height    = tensor->attr.size[1];
    vx_uint32 channel   = tensor->attr.size[2];
    bmpData = vnn_ReadJpegImage(image_name, width, height, channel);
    _CHECK_PTR(bmpData, final);

    /* image data transpose */
    vnn_ImagedataTranspose(bmpData, reorder, width, height, channel);

    /* handle mean-value and convert data to vx_dtype */
    data = vnn_ImagedataToDtype(tensor, bmpData, mean_value, width, height, channel);
    _CHECK_PTR(data, final);
    }
else if(fileType == NN_FILE_TENSOR)
    {
    data = vnn_ReadTensorImage(tensor, image_name, tensor->attr.size, tensor->attr.dim_num);
    _CHECK_PTR(data, final);
    }
else
    {
    printf("This Neural Network Only support tensor file or JPG image file\n");
    status = VX_FAILURE;
    _CHECK_STATUS(status, final);
    }

/* Copy the Pre-processed data to input tensor */
status = vsi_nn_CopyDataToTensor(graph, tensor, (vx_uint8 *)data);
_CHECK_STATUS(status, final);

/* Save the image data to txt */
vsi_nn_SaveTensorToTextByFp32( graph, tensor, "input.txt", NULL );

final:
if(bmpData)free(bmpData);
if(data)free(data);
return status;
}
