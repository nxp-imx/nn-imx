#ifndef __OVXLIB_COMPILATION_H__
#define __OVXLIB_COMPILATION_H__

#include "vsi_nn_pub.h"
#include "model.h"
#include "prepared_model.h"

namespace ovxlib
{
using shared_context = std::shared_ptr< std::pointer_traits<vsi_nn_context_t>::element_type >;
class Compilation
{
    public:
        Compilation(Model* model);

        virtual ~Compilation();

        virtual int run();

        virtual int finish() {return 0;};

        Model* getModel() {return model_;}


        PreparedModelPtr attachPreparedModel();

        void detachPreparedModel(PreparedModelPtr& prepared_model);

        int prepareModel();

    private:
        void cachePreparedModel(PreparedModelPtr& model);

        Model* model_;
        uint32_t prepared_model_cache_size_;
        std::map<std::string, PreparedModelPtr> prepared_models_;
        shared_context context_;
};
}

#endif
