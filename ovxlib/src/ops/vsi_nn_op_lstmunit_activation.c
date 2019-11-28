/****************************************************************************
*
*    Copyright (c) 2019 Vivante Corporation
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
#include <string.h>
#include <stdlib.h>

#include "vsi_nn_types.h"
#include "vsi_nn_platform.h"
#include "vsi_nn_log.h"
#include "vsi_nn_graph.h"
#include "vsi_nn_node.h"
#include "vsi_nn_prv.h"
#include "utils/vsi_nn_math.h"
#include "utils/vsi_nn_tensor_op.h"
#include "vsi_nn_ops.h"
#include "vsi_nn_tensor.h"
#include "vsi_nn_tensor_util.h"
#include "client/vsi_nn_vxkernel.h"
#include "utils/vsi_nn_util.h"

#define _ARG_NUM            (1)
#define _INPUT_NUM          (LSTMUNIT_ACT_INPUTS_COUNT)
#define _OUTPUT_NUM         (LSTMUNIT_ACT_OUTUTS_COUNT)
#define _IO_NUM             (_INPUT_NUM + _OUTPUT_NUM)
#define _PARAM_NUM          (_ARG_NUM + _IO_NUM)

extern vx_kernel_description_t * vx_kernel_LSTMUNIT_ACTIVATION_list[];

typedef enum _LSTMUNIT_nn_activation_type_e
{
    SIGMOID = VSI_NN_ACT_SIGMOID,
    HARD_SIGMOID = VSI_NN_ACT_HARD_SIGMOID,
}LSTMUNIT_nn_activation_type_e;

#define GEN_LSTMUNIT_KEY(_is_ln, _is_cifg, _is_proj, _is_hybrid, _is_peephole, \
_input_type, _output_type, _cell_type, _rec_act) \
((_is_ln << 31) | (_is_cifg << 30) | (_is_proj << 29) | (_is_hybrid << 28) | (_is_peephole << 27) \
| (_input_type << 23) | (_output_type << 19) | (_cell_type << 15) | (_rec_act << 10))

#define GEN_LSTMUNIT_KERNEL_SOURCE_NAME(_ln_cifg_proj_hybrid_, _input_type) \
    "vsi_nn_kernel_lstmunit_activation_"#_ln_cifg_proj_hybrid_"_"#_input_type

#define GEN_LSTMUNIT_STRUCT_ITEMS(_is_ln, _is_cifg, _is_proj, _is_hybrid, _is_peephole, _input_type, _output_type, \
_cell_type, _rec_act, _ln_cifg_proj_hybrid_) \
    GEN_LSTMUNIT_KEY(_is_ln, _is_cifg, _is_proj, _is_hybrid, _is_peephole, \
        _input_type, _output_type, _cell_type, _rec_act), \
    LSTMUNIT_SH_KERNEL_IDX(_ln_cifg_proj_hybrid_, _input_type, _output_type, _cell_type, _rec_act) \
    GEN_LSTMUNIT_KERNEL_SOURCE_NAME(_ln_cifg_proj_hybrid_, _input_type)

static struct {
        uint32_t key;
        uint32_t kernel_index;
        char *resource_name;
    } map[] =
    {
        /* layer norm + cifg + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, F16, F32, SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, F16, F16, SIGMOID, CLP)},
        /* layer norm + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, F16, F32, SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, F16, F16, SIGMOID, LP)},
        /* layer norm + cifg */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, F16, F16, SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, I16, F16, SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, U8,  F16, SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, I8,  F16, SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, F16, F32, SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, I16, F32, SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, U8,  F32, SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, I8,  F32, SIGMOID, CL)},
        /* layer norm */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, F16, F16, SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, I16, F16, SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, U8,  F16, SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, I8,  F16, SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, F16, F32, SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, I16, F32, SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, U8,  F32, SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, I8,  F32, SIGMOID, L)},
        /* layer norm + cifg + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, I16, F32, SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, I8,  F32, SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, U8,  F32, SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, I16, F16, SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, I8,  F16, SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, U8,  F16, SIGMOID, CLP)},
        /* layer norm + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, I16, F32, SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, I8,  F32, SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, U8,  F32, SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, I16, F16, SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, I8,  F16, SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, U8,  F16, SIGMOID, LP)},
        /* hybrid + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, F16, F32, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, F16, F16, SIGMOID, BP)},
        /* hybrid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, F16, F16, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, I16, F16, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, U8,  F16, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, I8,  F16, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, F16, F32, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, I16, F32, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, U8,  F32, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, I8,  F32, SIGMOID, B)},
        /* cifg + hybrid + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, F16, F32, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, F16, F16, SIGMOID, CBP)},
        /* cifg + hybrid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, F16, F16, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, I16, F16, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, U8,  F16, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, I8,  F16, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, F16, F32, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, I16, F32, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, U8,  F32, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, I8,  F32, SIGMOID, CB)},
        /* cifg + hybrid + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, I16, F32, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, I8,  F32, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, U8,  F32, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, I16, F16, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, I8,  F16, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, U8,  F16, SIGMOID, CBP)},
        /* hybrid + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, I16, F32, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, I8,  F32, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, U8,  F32, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, I16, F16, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, I8,  F16, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, U8,  F16, SIGMOID, BP)},
        /* hybrid + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  F16, F32, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  F16, F16, SIGMOID, BP)},
        /* hybrid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, U8,  F16, F16, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, U8,  U8,  F16, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, U8,  F16, F32, SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, U8,  U8,  F32, SIGMOID, B)},
        /* cifg + hybrid + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  F16, F32, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  F16, F16, SIGMOID, CBP)},
        /* cifg + hybrid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, U8,  F16, F16, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, U8,  U8,  F16, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, U8,  F16, F32, SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, U8,  U8,  F32, SIGMOID, CB)},
        /* cifg + hybrid + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  I16, F32, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  I8,  F32, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  U8,  F32, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  I16, F16, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  I8,  F16, SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  U8,  F16, SIGMOID, CBP)},
        /* hybrid + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  I16, F32, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  I8,  F32, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  U8,  F32, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  I16, F16, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  I8,  F16, SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  U8,  F16, SIGMOID, BP)},

        /* standard + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, F16, F32, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, F16, F16, SIGMOID, SP)},
        /* standard */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, F16, F16, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, I16, F16, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, U8,  F16, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, I8,  F16, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, F16, F32, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, I16, F32, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, U8,  F32, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, I8,  F32, SIGMOID, S)},
        /* cifg + standard + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, F16, F32, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, F16, F16, SIGMOID, CSP)},
        /* cifg + standard */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, F16, F16, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, I16, F16, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, U8,  F16, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, I8,  F16, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, F16, F32, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, I16, F32, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, U8,  F32, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, I8,  F32, SIGMOID, CS)},
        /* cifg + standard + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, I16, F32, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, I8,  F32, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, U8,  F32, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, I16, F16, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, I8,  F16, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, U8,  F16, SIGMOID, CSP)},
        /* standard + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, I16, F32, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, I8,  F32, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, U8,  F32, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, I16, F16, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, I8,  F16, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, U8,  F16, SIGMOID, SP)},
        /* standard + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  F16, F32, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  F16, F16, SIGMOID, SP)},
        /* standard */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, U8,  F16, F16, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, U8,  U8,  F16, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, U8,  F16, F32, SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, U8,  U8,  F32, SIGMOID, S)},
        /* cifg + standard + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  F16, F32, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  F16, F16, SIGMOID, CSP)},
        /* cifg + standard */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, U8,  F16, F16, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, U8,  U8,  F16, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, U8,  F16, F32, SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, U8,  U8,  F32, SIGMOID, CS)},
        /* cifg + standard + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  I16, F32, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  I8,  F32, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  U8,  F32, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  I16, F16, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  I8,  F16, SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  U8,  F16, SIGMOID, CSP)},
        /* standard + projection */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  I16, F32, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  I8,  F32, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  U8,  F32, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  I16, F16, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  I8,  F16, SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  U8,  F16, SIGMOID, SP)},
        /* layer norm + cifg + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, F16, F32, HARD_SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, F16, F16, HARD_SIGMOID, CLP)},
        /* layer norm + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, F16, F32, HARD_SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, F16, F16, HARD_SIGMOID, LP)},
        /* layer norm + cifg + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, F16, F16, HARD_SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, I16, F16, HARD_SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, U8,  F16, HARD_SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, I8,  F16, HARD_SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, F16, F32, HARD_SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, I16, F32, HARD_SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, U8,  F32, HARD_SIGMOID, CL)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 0, 0, 0, F16, I8,  F32, HARD_SIGMOID, CL)},
        /* layer norm + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, F16, F16, HARD_SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, I16, F16, HARD_SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, U8,  F16, HARD_SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, I8,  F16, HARD_SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, F16, F32, HARD_SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, I16, F32, HARD_SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, U8,  F32, HARD_SIGMOID, L)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 0, 0, 0, F16, I8,  F32, HARD_SIGMOID, L)},
        /* layer norm + cifg + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, I16, F32, HARD_SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, I8,  F32, HARD_SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, U8,  F32, HARD_SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, I16, F16, HARD_SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, I8,  F16, HARD_SIGMOID, CLP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 1, 1, 0, 0, F16, U8,  F16, HARD_SIGMOID, CLP)},
        /* layer norm + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, I16, F32, HARD_SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, I8,  F32, HARD_SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, U8,  F32, HARD_SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, I16, F16, HARD_SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, I8,  F16, HARD_SIGMOID, LP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(1, 0, 1, 0, 0, F16, U8,  F16, HARD_SIGMOID, LP)},
        /* hybrid + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, F16, F32, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, F16, F16, HARD_SIGMOID, BP)},
        /* hybrid + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, F16, F16, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, I16, F16, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, U8,  F16, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, I8,  F16, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, F16, F32, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, I16, F32, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, U8,  F32, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, F16, I8,  F32, HARD_SIGMOID, B)},
        /* cifg + hybrid + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, F16, F32, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, F16, F16, HARD_SIGMOID, CBP)},
        /* cifg + hybrid + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, F16, F16, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, I16, F16, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, U8,  F16, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, I8,  F16, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, F16, F32, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, I16, F32, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, U8,  F32, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, F16, I8,  F32, HARD_SIGMOID, CB)},
        /* cifg + hybrid + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, I16, F32, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, I8,  F32, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, U8,  F32, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, I16, F16, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, I8,  F16, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, F16, U8,  F16, HARD_SIGMOID, CBP)},
        /* hybrid + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, I16, F32, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, I8,  F32, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, U8,  F32, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, I16, F16, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, I8,  F16, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, F16, U8,  F16, HARD_SIGMOID, BP)},
        /* hybrid + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  F16, F32, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  F16, F16, HARD_SIGMOID, BP)},
        /* hybrid + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, U8,  F16, F16, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, U8,  U8,  F16, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, U8,  F16, F32, HARD_SIGMOID, B)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 1, 0, U8,  U8,  F32, HARD_SIGMOID, B)},
        /* cifg + hybrid + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  F16, F32, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  F16, F16, HARD_SIGMOID, CBP)},
        /* cifg + hybrid + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, U8,  F16, F16, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, U8,  U8,  F16, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, U8,  F16, F32, HARD_SIGMOID, CB)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 1, 0, U8,  U8,  F32, HARD_SIGMOID, CB)},
        /* cifg + hybrid + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  I16, F32, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  I8,  F32, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  U8,  F32, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  I16, F16, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  I8,  F16, HARD_SIGMOID, CBP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 1, 0, U8,  U8,  F16, HARD_SIGMOID, CBP)},
        /* hybrid + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  I16, F32, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  I8,  F32, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  U8,  F32, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  I16, F16, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  I8,  F16, HARD_SIGMOID, BP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 1, 0, U8,  U8,  F16, HARD_SIGMOID, BP)},

        /* standard + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, F16, F32, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, F16, F16, HARD_SIGMOID, SP)},
        /* standard + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, F16, F16, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, I16, F16, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, U8,  F16, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, I8,  F16, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, F16, F32, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, I16, F32, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, U8,  F32, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, F16, I8,  F32, HARD_SIGMOID, S)},
        /* cifg + standard + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, F16, F32, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, F16, F16, HARD_SIGMOID, CSP)},
        /* cifg + standard + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, F16, F16, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, I16, F16, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, U8,  F16, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, I8,  F16, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, F16, F32, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, I16, F32, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, U8,  F32, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, F16, I8,  F32, HARD_SIGMOID, CS)},
        /* cifg + standard + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, I16, F32, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, I8,  F32, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, U8,  F32, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, I16, F16, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, I8,  F16, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, F16, U8,  F16, HARD_SIGMOID, CSP)},
        /* standard + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, I16, F32, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, I8,  F32, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, U8,  F32, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, I16, F16, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, I8,  F16, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, F16, U8,  F16, HARD_SIGMOID, SP)},
        /* standard + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  F16, F32, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  F16, F16, HARD_SIGMOID, SP)},
        /* standard + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, U8,  F16, F16, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, U8,  U8,  F16, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, U8,  F16, F32, HARD_SIGMOID, S)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 0, 0, 0, U8,  U8,  F32, HARD_SIGMOID, S)},
        /* cifg + standard + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  F16, F32, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  F16, F16, HARD_SIGMOID, CSP)},
        /* cifg + standard + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, U8,  F16, F16, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, U8,  U8,  F16, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, U8,  F16, F32, HARD_SIGMOID, CS)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 0, 0, 0, U8,  U8,  F32, HARD_SIGMOID, CS)},
        /* cifg + standard + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  I16, F32, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  I8,  F32, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  U8,  F32, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  I16, F16, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  I8,  F16, HARD_SIGMOID, CSP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 1, 1, 0, 0, U8,  U8,  F16, HARD_SIGMOID, CSP)},
        /* standard + projection + hard_sigmoid */
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  I16, F32, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  I8,  F32, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  U8,  F32, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  I16, F16, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  I8,  F16, HARD_SIGMOID, SP)},
        {GEN_LSTMUNIT_STRUCT_ITEMS(0, 0, 1, 0, 0, U8,  U8,  F16, HARD_SIGMOID, SP)},
    };


uint32_t _set_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++)
    {
        if (inputs[i])
        {
            params[cnt] = (vx_reference)inputs[i]->t;
            cnt ++;
        }
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++ )
    {
        if (outputs[i])
        {
            params[cnt] = (vx_reference)outputs[i]->t;
            cnt ++;
        }
    }

    return cnt;
} /* _set_inputs_outputs() */


static void _set_sw_inputs_outputs
    (
    vx_reference * params,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    uint32_t i;
    uint32_t cnt;
    vsi_status status;

    /* Set inputs */
    cnt = 0;
    for( i = 0; i < _INPUT_NUM; i ++, cnt ++)
    {
        if (inputs[i])
        {
            /* Support high precision for inputs */
            if( NULL != inputs[i] && VSI_NN_TYPE_FLOAT32 == inputs[i]->attr.dtype.vx_type )
            {
                status = vsi_nn_SetTensorAttr(inputs[i], VSI_NN_TENSOR_ATTR_HIGH_PRECISION);
                if(VSI_SUCCESS != status)
                {
                    VSILOGE("Set tensor attr of inputs[%d] to high presision fail", i);
                }
            }

            params[cnt] = (vx_reference)inputs[i]->t;
        }
        else
            params[cnt] = NULL;
    }

    /* Set outputs */
    for( i = 0; i < _OUTPUT_NUM; i ++, cnt ++ )
    {
        if (outputs[i])
        {
            status = vsi_nn_SetTensorAttr(outputs[i], VSI_NN_TENSOR_ATTR_HIGH_PRECISION);
            if(VSI_SUCCESS != status)
            {
                VSILOGE("Set tensor attr of outputs[%d] to high presision fail", i);
            }
            params[cnt] = (vx_reference)outputs[i]->t;
        }
        else
            params[cnt] = NULL;
    }

} /* _set_inputs_outputs() */

#if 0
static vsi_status _create_params
    (
    vsi_nn_node_t * node,
    vx_reference * params,
    uint32_t num
    )
{
    vsi_status status;
    vx_context ctx;
    vsi_nn_lstmunit_activation_param * p;
    if( 0 == num )
    {
        return VSI_SUCCESS;
    }
    memset( params, 0, sizeof( vx_reference * ) * num );
    p = &(node->nn_param.lstmunit_activation);
    ctx = vxGetContext( (vx_reference)node->graph->g );
    /* Init parameters */
    #define _SET_PARAM( i, type, arg ) do{ \
        params[i] = (vx_reference)vxCreateScalar( ctx, type, &p->arg ); \
        status = vxGetStatus( params[i] ); \
        if( VSI_SUCCESS != status ) { \
            goto set_param_error; \
            } \
        } while(0)
    _SET_PARAM( 0, VX_TYPE_FLOAT32, forget_bias );
    _SET_PARAM( 1, VX_TYPE_FLOAT32, cell_clip );
    #undef _SET_PARAM

set_param_error:

    return status;
} /* _create_params */

static void _release_params
    (
    vx_reference * params,
    uint32_t num
    )
{
    uint32_t i;
    vx_scalar scalar;
    for( i = 0; i < num; i ++ )
    {
        scalar = (vx_scalar)params[i];
        vxReleaseScalar( &scalar );
    }
} /* _release_params() */
#endif

static vsi_status cpu_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    uint32_t param_num = _IO_NUM;
    vsi_nn_tensor_t * lstmunit_param = NULL;
    vsi_nn_tensor_attr_t attr;
    vsi_nn_lstmunit_activation_param * p;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.lstmunit_activation);

    /* Set inputs and outputs */
    _set_sw_inputs_outputs( params, inputs, outputs );

    memset(&attr, 0, sizeof(attr));
    attr.vtl = FALSE;
    attr.dim_num = 2;
    attr.size[0] = sizeof(vsi_nn_lstmunit_activation_param) / sizeof(uint8_t);
    attr.size[1] = 1;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;

    lstmunit_param = vsi_nn_CreateTensor(self->graph, &attr);
    p->local.lstmunit_param = lstmunit_param;

    vsi_nn_CopyDataToTensor(self->graph, lstmunit_param, (uint8_t*)p);
    /* Init parameters. */
    params[param_num] = (vx_reference)lstmunit_param->t;

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, param_num + 1 );

    return status;
}

static vsi_nn_shader_kernel_type_e get_lstm_unit_intra_type(vsi_nn_type_e type)
{
    switch (type)
    {
    case VSI_NN_TYPE_INT8:
        return I8;
    case VSI_NN_TYPE_INT16:
        return I16;
    case VSI_NN_TYPE_INT32:
        return I32;
    case VSI_NN_TYPE_INT64:
        return I64;
    case VSI_NN_TYPE_UINT8:
        return U8;
    case VSI_NN_TYPE_UINT16:
        return U16;
    case VSI_NN_TYPE_UINT32:
        return U32;
    case VSI_NN_TYPE_FLOAT16:
        return F16;
    case VSI_NN_TYPE_FLOAT32:
        return F32;
    default:
        VSILOGE("error data type %d", type);
        break;
    }

    return I8;
}

static void _get_lstmunit_hashtable_idx
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_type_e inputFormat = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.dtype.vx_type;
    vsi_nn_type_e cellFormat = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.dtype.vx_type;
    vsi_nn_type_e outputFormat  = outputs[0]->attr.dtype.vx_type;
    vsi_nn_shader_kernel_type_e _input_type;
    vsi_nn_shader_kernel_type_e _output_type;
    vsi_nn_shader_kernel_type_e _cell_type;
    uint32_t key;
    uint32_t _is_ln= 0;
    uint32_t _is_cifg= 0;
    uint32_t _is_proj= 0;
    uint32_t _is_hybrid= 0;
    uint32_t _is_peephole= 0;
    uint32_t i = 0;

    vsi_nn_lstmunit_activation_param * p;

    p = &(self->nn_param.lstmunit_activation);

    _is_ln = p->is_layer_norm ? 1 : 0;
    _is_cifg = p->is_cifg ? 1 : 0;
    _is_proj = p->is_projection ? 1 : 0;
    _is_hybrid = p->is_hybrid ? 1 : 0;
    _is_peephole = p->is_peephole ? 1 : 0;

    _input_type = get_lstm_unit_intra_type(inputFormat);
    _output_type = get_lstm_unit_intra_type(outputFormat);
    _cell_type = get_lstm_unit_intra_type(cellFormat);

    key = GEN_LSTMUNIT_KEY(_is_ln, _is_cifg, _is_proj, _is_hybrid, _is_peephole,
        _input_type, _output_type, _cell_type, p->recurrent_activation);

    for (i = 0; i < sizeof(map) / sizeof(map[0]); i++)
    {
        if (key == map[i].key)
        {
            p->local.hash_idx = i;
            p->local.execute_on_sw = FALSE;
            return;
        }
    }

    p->local.execute_on_sw = TRUE;
    VSILOGE("Not support data format or feature![LSTMUNIT_ACTIVATION]\n");
}

static vsi_bool vx_op_pre_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs,
    vsi_nn_kernel_info_t * kernel_info
    )
{
    vsi_nn_lstmunit_activation_param * p;

    p = &(self->nn_param.lstmunit_activation);

    kernel_info->kernel_index = map[p->local.hash_idx].kernel_index;
    kernel_info->resource_name[0] = map[p->local.hash_idx].resource_name;

    return TRUE;
}

static vsi_status vx_op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status = VSI_SUCCESS;
    vx_reference params[_PARAM_NUM];
    vx_border_t border;
    vsi_nn_tensor_t * lstmunit_param = NULL;
    vsi_nn_tensor_attr_t attr;
    uint32_t param_num = 0;
    vsi_nn_lstmunit_activation_param * p;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    if( NULL == self->n )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.lstmunit_activation);

    /* Set inputs and outputs */
    param_num = _set_inputs_outputs( params, inputs, outputs );
    /*TODO: Add code if need to change your parameter*/

    memset(&attr, 0, sizeof(attr));
    attr.vtl = FALSE;
    attr.dim_num = 2;
    attr.size[0] = sizeof(vsi_nn_lstmunit_activation_param) / sizeof(uint8_t);
    attr.size[1] = 1;
    attr.dtype.vx_type = VSI_NN_TYPE_UINT8;

    lstmunit_param = vsi_nn_CreateTensor(self->graph, &attr);

    vsi_nn_CopyDataToTensor(self->graph, lstmunit_param, (uint8_t*)p);
    /* Init parameters. */
    //_create_params( self, args, _ARG_NUM );
    params[param_num] = (vx_reference)lstmunit_param->t;

    /* Pass parameters to node. */
    status = vsi_nn_ClientNodePassParameters( self->n, params, param_num + 1 );

    //_release_params( args, _ARG_NUM );

    border.mode = VX_BORDER_REPLICATE;
    status |= vxSetNodeAttribute(self->n, VX_NODE_BORDER, &border, sizeof(border));

    if (lstmunit_param) vsi_nn_ReleaseTensor(&lstmunit_param);

    return status;
}

static vsi_nn_op_compute_t op_compute_list[] =
{
    cpu_op_compute,
    vx_op_compute,
    NULL
};

static vsi_status op_compute
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_status status;
    vsi_nn_kernel_info_t kernel_info;
    vsi_nn_lstmunit_activation_param * p = NULL;

    memset(&kernel_info, 0x0, sizeof(vsi_nn_kernel_info_t));
    status = VSI_FAILURE;
    if( NULL == self )
    {
        return VSI_FAILURE;
    }

    p = &(self->nn_param.lstmunit_activation);

   _get_lstmunit_hashtable_idx(self, inputs, outputs);

   if (p->local.execute_on_sw || !vsi_nn_IsEVISFeatureAvaiable(self->graph->ctx))
    {
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.resource_name[0] = "vsi_nn_kernel_lstmunit_activation";
        kernel_info.type = VX_KERNEL_TYPE_CPU;
        kernel_info.kernel = vx_kernel_LSTMUNIT_ACTIVATION_list;
        kernel_info.init_index = 0;

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
        self->graph, &kernel_info);
        if (kernel_info.resource_name) free(kernel_info.resource_name);
        if( NULL == self->n )
        {
            return VSI_FAILURE;
        }

        status = cpu_op_compute(self, inputs, outputs);
    }
    else
    {
        kernel_info.type = vsi_nn_GetVXKernelTypeForShader();
        kernel_info.kernel = vx_kernel_LSTMUNIT_ACTIVATION_list;
        kernel_info.resource_num = 1;
        kernel_info.resource_name = (char **)malloc(kernel_info.resource_num * sizeof(char *));
        kernel_info.init_index = 1;
        kernel_info.resource_name[0] = "vsi_nn_kernel_lstmunit_activation_clp";

        if (vsi_nn_is_do_vx_op_pre_init(kernel_info.type))
        {
            vx_op_pre_compute(self, inputs, outputs, &kernel_info);
        }

        self->n = vsi_nn_RegisterClientKernelAndNewNode(
                self->graph, &kernel_info);
        if (kernel_info.resource_name)
        {
            free(kernel_info.resource_name);
        }
        if( NULL == self->n )
        {
            return VSI_FAILURE;
        }
        if (NULL != op_compute_list[kernel_info.init_index])
        {
            status = op_compute_list[kernel_info.init_index](self, inputs, outputs);
        }
    }

    return status;
} /* op_compute() */

static vsi_bool op_check
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    /*TODO: Check tensor shapes. */
    return TRUE;
} /* op_check() */

static vsi_bool op_setup
    (
    vsi_nn_node_t * self,
    vsi_nn_tensor_t ** inputs,
    vsi_nn_tensor_t ** outputs
    )
{
    vsi_nn_lstmunit_activation_param * p;
    vsi_nn_dtype_t dst_dtype;
    int32_t ifco_start_index = 0;
    vsi_nn_tensor_attr_t attr;
    int32_t i = 0;

    memset(&attr, 0, sizeof(vsi_nn_tensor_attr_t));
    if( NULL == self )
    {
        return FALSE;
    }

    p = &(self->nn_param.lstmunit_activation);

    p->is_cifg = inputs[LSTMUNIT_ACT_INPUT_FC_I] == NULL;
    p->is_projection = outputs[LSTMUNIT_ACT_HSTATE_OUT] == NULL;
    p->is_layer_norm = inputs[LSTMUNIT_ACT_LN_WF] != NULL;
    p->is_hybrid = p->is_layer_norm ? 0 : inputs[LSTMUNIT_ACT_DATA_BF] != NULL;
    p->recurrent_activation = p->recurrent_activation == VSI_NN_ACT_NONE ?
        VSI_NN_ACT_SIGMOID : p->recurrent_activation;

    for( i = ifco_start_index; i < 4; i++ )
    {
        vsi_nn_tensor_t* t0 = NULL;
        vsi_nn_tensor_t* t1 = NULL;
        dst_dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;
        dst_dtype.vx_type = VSI_NN_TYPE_FLOAT32;

        if (inputs[LSTMUNIT_ACT_DATA_BI + i] && inputs[LSTMUNIT_ACT_DATA_BI + i]->attr.dim_num == 1)
        {
            memcpy(&attr, &(inputs[LSTMUNIT_ACT_DATA_BI + i]->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            t0 = vsi_nn_CreateTensor( self->graph, &attr );
            vsi_nn_ReshapeTensor(self->graph, inputs[LSTMUNIT_ACT_DATA_BI + i], t0, attr.size, attr.dim_num);

            if( dst_dtype.vx_type != t0->attr.dtype.vx_type
                && dst_dtype.qnt_type != t0->attr.dtype.qnt_type )
            {
                p->local.tensors[LSTMUNIT_ACT_TENSOR_BI + i] =
                    vsi_nn_ConvertTensorDtype( self->graph, t0, &dst_dtype );
                vsi_nn_ReleaseTensor( &t0 );
            }
            else
            {
                p->local.tensors[LSTMUNIT_ACT_TENSOR_BI + i] = t0;
            }

            inputs[LSTMUNIT_ACT_DATA_BI + i] = p->local.tensors[LSTMUNIT_ACT_TENSOR_BI + i];
        }

        if (inputs[LSTMUNIT_ACT_LN_WI + i] && inputs[LSTMUNIT_ACT_LN_WI + i]->attr.dim_num == 1)
        {
            memcpy(&attr, &(inputs[LSTMUNIT_ACT_LN_WI + i]->attr), sizeof(vsi_nn_tensor_attr_t));
            attr.size[1] = 1;
            attr.dim_num = 2;
            t1 = vsi_nn_CreateTensor( self->graph, &attr );
            vsi_nn_ReshapeTensor(self->graph, inputs[LSTMUNIT_ACT_LN_WI + i], t1, attr.size, attr.dim_num);

            if( dst_dtype.vx_type != t1->attr.dtype.vx_type
                && dst_dtype.qnt_type != t1->attr.dtype.qnt_type )
            {
                p->local.tensors[LSTMUNIT_ACT_TENSOR_LN_WI + i] =
                    vsi_nn_ConvertTensorDtype( self->graph, t1, &dst_dtype );
                vsi_nn_ReleaseTensor( &t1 );
            }
            else
            {
                p->local.tensors[LSTMUNIT_ACT_TENSOR_LN_WI + i] = t1;
            }

            inputs[LSTMUNIT_ACT_LN_WI + i] = p->local.tensors[LSTMUNIT_ACT_TENSOR_LN_WI + i];
        }
    }

    if( VSI_NN_DIM_AUTO == outputs[LSTMUNIT_ACT_OUTPUT]->attr.dim_num )
    {
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.dim_num = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.dim_num;
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[0] = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.size[0];
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[1] = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.size[1];
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[2] = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.size[2];
        outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[3] = inputs[LSTMUNIT_ACT_INPUT_FC_F]->attr.size[3];
    }

    if( VSI_NN_DIM_AUTO == outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.dim_num )
    {
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.dim_num = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.dim_num;
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.size[0] = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.size[0];
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.size[1] = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.size[1];
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.size[2] = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.size[2];
        outputs[LSTMUNIT_ACT_CSTATE_OUT]->attr.size[3] = inputs[LSTMUNIT_ACT_CSTATE_IN]->attr.size[3];
    }

    if (outputs[LSTMUNIT_ACT_HSTATE_OUT] && VSI_NN_DIM_AUTO == outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.dim_num )
    {
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.dim_num = outputs[LSTMUNIT_ACT_OUTPUT]->attr.dim_num;
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.size[0] = outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[0];
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.size[1] = outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[1];
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.size[2] = outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[2];
        outputs[LSTMUNIT_ACT_HSTATE_OUT]->attr.size[3] = outputs[LSTMUNIT_ACT_OUTPUT]->attr.size[3];
    }

    return TRUE;
} /* op_setup() */

static vsi_status op_deinit
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;
    int32_t i = 0;

    for (i = 0; i < LSTMUNIT_ACT_TENSOR_CNT; i++)
    {
        if (self->nn_param.lstmunit_activation.local.tensors[i] != NULL)
        {
            vsi_nn_ReleaseTensor(&self->nn_param.lstmunit_activation.local.tensors[i]);
            self->nn_param.lstmunit_activation.local.tensors[i] = NULL;
        }
    }

    if(self->nn_param.lstmunit_activation.local.lstmunit_param != NULL)
    {
        vsi_nn_ReleaseTensor(&self->nn_param.lstmunit_activation.local.lstmunit_param);
        self->nn_param.lstmunit_activation.local.lstmunit_param = NULL;
    }

    return status;
} /* op_deinit() */

static vsi_status op_init
    (
    vsi_nn_node_t * self
    )
{
    vsi_status status = VSI_SUCCESS;

    self->nn_param.lstmunit_activation.recurrent_activation = VSI_NN_ACT_SIGMOID;

    return status;
} /* op_init() */

#ifdef __cpluplus
extern "C" {
#endif
/* Registrar */
DEF_OP_REG
    (
    /* op_name    */ LSTMUNIT_ACTIVATION,
    /* init       */ op_init,
    /* compute    */ op_compute,
    /* deinit     */ op_deinit,
    /* check      */ op_check,
    /* setup      */ op_setup,
    /* optimize   */ NULL,
    /* input_num  */ _INPUT_NUM,
    /* output_num */ _OUTPUT_NUM
    );
#ifdef __cpluplus
}
#endif
