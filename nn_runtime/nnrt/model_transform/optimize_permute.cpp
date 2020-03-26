#include "nnrt/model.hpp"
#include "nnrt/error.hpp"
#include "nnrt/model_transform/transformations.hpp"

namespace nnrt
{
int OptimizePermute::run(Model * model, bool * modified)
{
    (void)model; // unused variable
    *modified = false;
    return NNA_ERROR_CODE(NO_ERROR);
#if 0
    *modified = false;
    if (nullptr == model)
    {
        return NNA_ERROR_CODE(NO_ERROR);
    }

    auto is_unused_permute = [](Operation * operation) -> bool
    {
        // TODO:
        return false;
    };

    // Move permute
    auto operations = model->operations();
    for (auto it = operations.begin(); it != operations.end(); ++ it)
    {
        //TODO:
    }

    //// Merge permute into params
    //for (auto it = operations.begin(); it != operations.end(); ++ it)
    //{
    //    Operation * operation = it->second;
    //    if (_merge_param_operations.find(operation->type) != _merge_param_operations.end())
    //    {
    //        *modified = true;
    //        _merge_param_operations[operation->type](model, operation);
    //    }
    //}

    // Remove unused permute
    std::vector<int> unused_permute;
    for (auto it = operations.begin(); it != operations.end(); ++ it)
    {
        if (is_unused_permute(it->second))
        {
            *modified = true;
            unused_permute.push_back(it->first);
        }
    }
    //for (auto it : unused_permute)
    //{
    //    // TODO: remove
    //}
    return NNA_ERROR_CODE(NO_ERROR);
#endif
}
}
