#include <assert.h>
#include "vsi_nn_pub.h"
#include "model.h"
#include "compilation.h"
#include "error.h"
//#include "graph_transformations/transformations.h"

namespace ovxlib
{
Compilation::Compilation(Model * model)
    : model_(model)
{
}

Compilation::~Compilation()
{

}

int Compilation::run()
{
    int err = AERROR_CODE(NO_ERROR);
    // This code is moved to prepare model
#if 0
    if (!model_->isCompiled()) {
        TransformationSet transformations;
        // NOTE: The transformation order is important.
        model_->echo();
        transformations.add(new NnApiInterpreter());
        transformations.add(new AlignBroadcastOp());
        transformations.add(new T2C());
        transformations.add(new OptimizePermute());
        transformations.add(new ValidateQuantizedGraph());

        // relaxed mode
        if (model_->isRelaxed())
            transformations.add(new Fp32ToFp16());

        err = transformations.once(model_);
        model_->freezeCompile();
        if (err != AERROR_CODE(NO_ERROR)) {
            return err;
        }
        model_->echo();
    }
#endif
    return err;
}

}
