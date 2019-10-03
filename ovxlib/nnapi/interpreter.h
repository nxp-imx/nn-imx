#ifndef __OVXLIB_INTERPRETER_H__
#define __OVXLIB_INTERPRETER_H__

#include "model.h"

namespace ovxlib
{
class Interpreter
{
    public:
        virtual ~Interpreter(){};

        virtual int run(Model * model, bool * modified) = 0;

        virtual const char* name() = 0;
};

};
#endif