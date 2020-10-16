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
#include "nnrt/logging.hpp"
#include "nnrt/event.hpp"
#include "nnrt/execution.hpp"
#include "nnrt/error.hpp"

namespace nnrt
{
Event::Event(int state)
    : state_(state) {}

Event::~Event() {}

void Event::notify(int state)
{
    std::unique_lock<std::mutex> lk(mutex_);
    state_ = state;
    condition_.notify_all();
}

int Event::wait()
{
    std::unique_lock<std::mutex> lk(mutex_);
    condition_.wait(lk, [this]{return state_ != IN_PROCESS;});
    int error = NNA_ERROR_CODE(NO_ERROR);
    switch (state_)
    {
        case IDLE:
            break;
        case IN_PROCESS:
            NNRT_LOGW_PRINT("Some errors occured?");
            break;
        case ERROR_OCCURED:
            error = NNA_ERROR_CODE(BAD_DATA);
            break;
        case CANCELED:
            //error = NNA_ERROR_CODE(ERROR_DELETED);
            break;
        case COMPLETED:
            break;
        default:
            NNRT_LOGW_PRINT("Got error state: %d", state_);
            break;

    }
    return error;
}
}
