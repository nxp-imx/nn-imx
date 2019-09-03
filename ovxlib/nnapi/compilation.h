#ifndef __OVXLIB_COMPILATION_H__
#define __OVXLIB_COMPILATION_H__

#include "vsi_nn_pub.h"
#include "model.h"

namespace ovxlib
{
class Compilation
{
    public:
        Compilation(Model* model);

        virtual ~Compilation();

        virtual int run();

        virtual int finish() {return 0;};

        Model* getModel() {return model_;}

    private:
        Model* model_;
};
}

#endif
