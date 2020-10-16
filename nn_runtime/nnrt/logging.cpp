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
#if defined(__ANDROID__)
#include <android/log.h>
#endif
#include "nnrt/logging.hpp"

namespace nnrt {
namespace logging {
    void Logger::print(const char* fmt, ...) {
        if(!enabled()) {
            return ;
        }

        #define MAX_MSG_LENGTH 1024
        char arg_buffer[MAX_MSG_LENGTH] = {0};
        va_list arg;
        va_start(arg, fmt);
        vsnprintf(arg_buffer, MAX_MSG_LENGTH, fmt, arg);
        va_end(arg);
#if defined(__ANDROID__)
        int priority = ANDROID_LOG_DEFAULT;
        switch (level_) {
            case Level::Debug:
                priority = ANDROID_LOG_DEBUG;
                break;
            case Level::Info:
                priority = ANDROID_LOG_INFO;
                break;
            case Level::Warn:
                priority = ANDROID_LOG_WARN;
                break;
            case Level::Error:
                priority = ANDROID_LOG_ERROR;
                break;
            default:
                break;
        }
        __android_log_print(priority, tag_.c_str(), "%s", arg_buffer);
#else
        std::string header = getHeader();
        fprintf(stderr, "%s%s\n", header.c_str(), arg_buffer);
#endif
    }
}
}

