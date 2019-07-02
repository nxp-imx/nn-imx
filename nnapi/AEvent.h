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


//
// Created by kg on 18-3-8.
//

#ifndef NNAPI_AEVENT_H
#define NNAPI_AEVENT_H

#include <VX/vx.h>
#include <memory>
#include "AExecution.h"


class AEvent{

public:
    explicit AEvent(AExecution *e): m_exe(e){};
    ~AEvent() {};

    int AEvent_wait();

private:
    AExecution *m_exe;
};

#endif //NNAPI_AEVENT_H
