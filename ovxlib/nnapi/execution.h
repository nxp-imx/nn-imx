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
using shared_context = std::shared_ptr< std::pointer_traits<vsi_nn_context_t>::element_type >;
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

        PreparedModel* getPreparedModel();

        void complete(int status, bool notify_event = false);

        int fillInput(PreparedModel* prepared_model);

        int fillOutput(PreparedModel* prepared_model);

    private:
        void notify(int code);

        PreparedModel* prepared_model_;
        std::vector<ExecutionIO*> inputs_;
        std::vector<ExecutionIO*> outputs_;
        Compilation* compilation_;
        bool running_;
        bool ask_for_quit_;
        std::mutex mutex_;
        std::vector<Event*> events_;
        shared_context local_context;
};
}
#endif
