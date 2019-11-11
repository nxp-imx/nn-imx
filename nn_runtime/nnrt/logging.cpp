#if defined(__ANDROID__)
#include <android/log.h>
#endif
#include "logging.hpp"

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

