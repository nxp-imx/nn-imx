#include <assert.h>
#include "vsi_nn_pub.h"
#include "model.h"
#include "compilation.h"
#include "error.h"
#include "prepared_model.h"
//#include "graph_transformations/transformations.h"

namespace ovxlib
{

shared_context global_context;

struct ContextDeleter {
    void operator()(vsi_nn_context_t ctx) {
    VSILOGD("Release context");
        vsi_nn_ReleaseContext(&ctx);
   }
};



Compilation::Compilation(Model * model)
    : model_(model)
    , prepared_model_cache_size_(1)
{
   if (!global_context) {
       VSILOGD("===== create share_ptr context ==========\n");
       global_context.reset(vsi_nn_CreateContext(), ContextDeleter());
   }
   context_ = global_context;
}

Compilation::~Compilation()
{
    prepared_models_.clear();
    context_.reset();
    if(global_context.use_count() == 1){
         global_context.reset();
    }
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

void Compilation::cachePreparedModel(PreparedModelPtr& prepared_model)
{
    if (prepared_models_.size() + 1 > prepared_model_cache_size_) {
        std::vector<std::string> keys_to_remove;
        for (auto it = prepared_models_.begin(); it != prepared_models_.end(); it ++) {
            if (it->second.use_count() == 1) {
                keys_to_remove.push_back(it->first);
            }
        }
        for (auto key : keys_to_remove) {
            prepared_models_.erase(key);
        }
    }
    prepared_models_[prepared_model->signature()] = prepared_model;
}

PreparedModelPtr Compilation::attachPreparedModel()
{
    std::string model_signature = model_->signature();
    auto it = prepared_models_.find(model_signature);
    PreparedModelPtr prepared_model = nullptr;
    if (it == prepared_models_.end()) {
        VSILOGD("Model signature not in cache, call prepareModel() first.");
        assert(false);
    } else {
        prepared_model = it->second;
    }
    return prepared_model;
}

int Compilation::prepareModel()
{
    int err = AERROR_CODE(NO_ERROR);
    std::string model_signature = model_->generateSignature();
    auto it = prepared_models_.find(model_signature);
    PreparedModelPtr prepared_model = nullptr;
    if (it == prepared_models_.end()) {
        prepared_model = std::make_shared<PreparedModel>(model_);
        err = prepared_model->prepare(context_.get());
        cachePreparedModel(prepared_model);
    }
    return err;
}

void Compilation::detachPreparedModel(PreparedModelPtr& prepared_model)
{
    std::string key = prepared_model->signature();
    auto it = prepared_models_.find(key);
    assert(it != prepared_models_.end());
    prepared_model.reset();
}

}
