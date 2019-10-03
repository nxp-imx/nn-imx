#ifndef __OVXLIB_EXECUTION_H__
#define __OVXLIB_EXECUTION_H__

#include <mutex>
#include <vector>
#include "model.h"
#include "operand.h"
#include "compilation.h"
#include "prepared_model.h"
#include "memory.h"
#include "event.h"

namespace ovxlib
{
struct ExecutionIO;

class Execution
{
    public:
        Execution(Compilation* compilation);
        virtual ~Execution();

        virtual bool isRunning() {return running_;}

        virtual void quit();

        virtual int startCompute(Event* event);

        virtual int compute();

        int setInput(uint32_t index, const Operand* operand_type,
                const void* buffer, size_t length);

        int setInputFromMemory(uint32_t index, const Operand* operand_type,
                const Memory* memory, size_t offset, size_t length);

        int setOutput(uint32_t index, const Operand* operand_type,
                void* buffer, size_t length);

        int setOutputFromMemory(uint32_t index, const Operand* operand_type,
                const Memory* memory, size_t offset, size_t length);

        PreparedModelPtr getPreparedModel();

        void complete(int status, bool notify_event = false);

        int fillInput(PreparedModelPtr prepared_model);

        int fillOutput(PreparedModelPtr prepared_model);

        Compilation* getCompilation() { return compilation_; }

    private:
        void notify(int code);

        std::vector<ExecutionIO*> inputs_;
        std::vector<ExecutionIO*> outputs_;
        Compilation* compilation_;
        bool running_;
        bool ask_for_quit_;
        std::mutex mutex_;
        Event* event_{nullptr};
};
}
#endif
