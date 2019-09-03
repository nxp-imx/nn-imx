#ifndef __OVXLIB_EVENT_H__
#define __OVXLIB_EVENT_H__

#include <mutex>
#include <condition_variable>

namespace ovxlib
{
class Event
{
    public:
        enum
        {
            IDLE,
            IN_PROCESS,
            ERROR_OCCURED,
            CANCELED,
            COMPLETED,
        };

        Event(int state = IDLE);
        ~Event();

        int wait();

        void notify(int state);

    private:
        int state_;
        std::mutex mutex_;
        std::condition_variable condition_;
};
}
#endif
