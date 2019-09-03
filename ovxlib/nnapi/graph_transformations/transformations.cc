#include <assert.h>
#include "model.h"
#include "error.h"
#include "graph_transformations/transformations.h"

namespace ovxlib
{
TransformationSet::~TransformationSet()
{
    for (auto trans : transformations_) {
        delete trans;
    }
}

int TransformationSet::run(Model* model)
{
    int err = AERROR_CODE(NO_ERROR);
    //bool modified = false;
    //TODO: Run tranformations until there is no modifications.
    assert(false);
    return err;
}

int TransformationSet::once(Model* model)
{
    int err = AERROR_CODE(NO_ERROR);
    bool modified = false;
    for (auto trans : transformations_) {
        VSILOGD("Run %s", trans->name());
        err = trans->run(model, &modified);
        if (AERROR_CODE(NO_ERROR) != err) {
            VSILOGW("Run %s fail.", trans->name());
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
