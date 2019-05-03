/****************************************************************************
*
*    Copyright (c) 2005 - 2019 by Vivante Corp.  All rights reserved.
*
*    The material in this file is confidential and contains trade secrets
*    of Vivante Corporation. This is proprietary information owned by
*    Vivante Corporation. No part of this work may be disclosed,
*    reproduced, copied, transmitted, or used in any way for any purpose,
*    without the express written permission of Vivante Corporation.
*
*****************************************************************************/


#ifndef _UTIL_H
#define _UTIL_H

#include <VX/vx.h>
#include "AModel.h"
#define CONVERT_TENSOR 1

extern vx_status convertDims(vx_uint32 * dims, vx_uint32 *org_dims, vx_uint32 count, bool SNforFC = false);

extern void convertTensorDataFromFp322Fp16(vx_context context,AnnOperand &operand);

extern void convertRankAndFormat(vx_context context,AnnOperand &operand, bool convertSNForFC = vx_false_e);

extern vx_tensor convertScalar2Tensor(vx_context context,AnnOperand &operand);

extern int vxcMemcpy(vx_context context, vx_tensor& tensor, void *hostPtr, vx_accessor_e usage);
#endif