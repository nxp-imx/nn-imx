#include "vsi_nn_pub.h"
#include "event.h"
#include "execution.h"
#include "error.h"

namespace ovxlib
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
    int error = AERROR_CODE(NO_ERROR);
    switch (state_)
    {
        case IDLE:
            break;
        case IN_PROCESS:
            VSILOGW("Some errors occured?");
            break;
        case ERROR_OCCURED:
            error = AERROR_CODE(BAD_DATA);
            break;
        case CANCELED:
            //error = AERROR_CODE(ERROR_DELETED);
            break;
        case COMPLETED:
            break;
        default:
            VSILOGW("Got error state: %d", state_);
            break;

    }
    return error;
}
}
