#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "NeuralNetworks.h"

int main(int argc, char** argv)
{
    std::vector<uint32_t> dimensions = {1,5};
    ANeuralNetworksOperandType operandType = {
        ANEURALNETWORKS_TENSOR_FLOAT32,
        (uint32_t)dimensions.size(),
        dimensions.data(),
        1,
        0,
    };
    float inputBuffer[] = {-2.0f,-1.0f,0.0f,1.0f,2.0f};
    float outputBuffer[5] = {NAN};


    for(uint32_t i = 0; i < 50000; i ++) {
        ANeuralNetworksModel* model;
        int err = ANeuralNetworksModel_create(&model);
        assert(err == ANEURALNETWORKS_NO_ERROR);

        uint32_t inputOperand = 0;
        uint32_t outputOperand = 1;
        err = ANeuralNetworksModel_addOperand(model, &operandType);
        assert(err == ANEURALNETWORKS_NO_ERROR);
        err = ANeuralNetworksModel_addOperand(model, &operandType);
        assert(err == ANEURALNETWORKS_NO_ERROR);
        err = ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_RELU,
                                1, &inputOperand, 1, &outputOperand);
        err = ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, &inputOperand,
                1, &outputOperand);
        assert(err == ANEURALNETWORKS_NO_ERROR);
        err = ANeuralNetworksModel_finish(model);
        assert(err == ANEURALNETWORKS_NO_ERROR);
        ANeuralNetworksCompilation* compilation;
        err = ANeuralNetworksCompilation_create(model, &compilation);
        assert(err == ANEURALNETWORKS_NO_ERROR);
        err = ANeuralNetworksCompilation_finish(compilation);
        assert(err == ANEURALNETWORKS_NO_ERROR);
        ANeuralNetworksExecution* execution;
        err = ANeuralNetworksExecution_create(compilation, &execution);
        assert(err == ANEURALNETWORKS_NO_ERROR);

        err = ANeuralNetworksExecution_setInput(execution, 0, NULL,
                inputBuffer, sizeof(inputBuffer));
        assert(err == ANEURALNETWORKS_NO_ERROR);
        err = ANeuralNetworksExecution_setOutput(execution, 0, NULL,
                outputBuffer, sizeof(outputBuffer));
        assert(err == ANEURALNETWORKS_NO_ERROR);
        ANeuralNetworksEvent* event;
        err = ANeuralNetworksExecution_startCompute(execution, &event);
        assert(err == ANEURALNETWORKS_NO_ERROR);
        err = ANeuralNetworksEvent_wait(event);
        assert(err == ANEURALNETWORKS_NO_ERROR);
        ANeuralNetworksEvent_free(event);
        ANeuralNetworksExecution_free(execution);
        ANeuralNetworksCompilation_free(compilation);
        ANeuralNetworksModel_free(model);
    }

    return 0;
}
