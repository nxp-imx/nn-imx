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


/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_OEM_H
#define ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_OEM_H

enum {
    /** OEM specific scalar value. */
    ANEURALNETWORKS_OEM_SCALAR = 10000,

    /** A tensor of OEM specific values. */
    ANEURALNETWORKS_TENSOR_OEM_BYTE = 10001,
};  // extends OperandCode

enum {
    /** OEM specific operation. */
    ANEURALNETWORKS_OEM_OPERATION = 10000,
};  // extends OperationCode


#endif  // ANDROID_ML_NN_RUNTIME_NEURAL_NETWORKS_OEM_H
