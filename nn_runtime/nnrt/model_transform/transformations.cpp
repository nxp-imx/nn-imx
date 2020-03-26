#include <cassert>

#include "nnrt/model.hpp"
#include "nnrt/error.hpp"
#include "nnrt/model_transform/transformations.hpp"

namespace nnrt
{
TransformationSet::~TransformationSet()
{
    for (auto trans : transformations_) {
        delete trans;
    }
}

int TransformationSet::run(Model* model)
{
    (void)model; // unused variable
    int err = NNA_ERROR_CODE(NO_ERROR);
    //bool modified = false;
    //TODO: Run tranformations until there is no modifications.
    assert(false);
    return err;
}

int TransformationSet::once(Model* model)
{
    int err = NNA_ERROR_CODE(NO_ERROR);
    bool modified = false;
    for (auto trans : transformations_) {
        NNRT_LOGD_PRINT("Run %s", trans->name());
        err = trans->run(model, &modified);
        if (NNA_ERROR_CODE(NO_ERROR) != err) {
            NNRT_LOGW_PRINT("Run %s fail.", trans->name());
            return err;
        }
    }
    return err;
}

void TransformationSet::add(Transformation* transformation)
{
    transformations_.push_back(transformation);
}
}
