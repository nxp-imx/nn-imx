#include "model.h"
#include "operand.h"
#include "graph_transformations/transformations.h"

using namespace ovxlib;

int main(int argc, char ** argv)
{
Model model;
Operand operand0 = {.type = OperandType::TENSOR_FLOAT32,
                    .dimensions = {1,512,256,3}, .numberOfConsumers = 1 };
Operand operand1 = {.type = OperandType::TENSOR_FLOAT32,
                    .dimensions = {1,256,128,16}, .numberOfConsumers = 1 };
Operand operand2 = {.type = OperandType::TENSOR_FLOAT32,
                    .dimensions = {1,64,64,16}, .numberOfConsumers = 0 };
Operand operand3 = {.type = OperandType::TENSOR_FLOAT32,
                    .dimensions = {1,3,3,7}, .numberOfConsumers = 0 };
Operand operand4 = {.type = OperandType::TENSOR_FLOAT32,
                    .dimensions = {1,50,25,256}, .numberOfConsumers = 0 };
Operand operand5 = {.type = OperandType::TENSOR_FLOAT32,
                    .dimensions = {1,50,26,128}, .numberOfConsumers = 0 };

Operation operation0 = {.type = OperationType::CONV_2D, .inputs = {0}, .outputs = {1}};
Operation operation1 = {.type = OperationType::MAX_POOL_2D, .inputs = {1}, .outputs = {2}};
Operation operation2 = {.type = OperationType::RESHAPE, .inputs = {2}, .outputs = {3}};
Operation operation3 = {.type = OperationType::ADD, .inputs = {1,3}, .outputs = {4}};
Operation operation4 = {.type = OperationType::RELU, .inputs = {4}, .outputs = {5}};

model.addOperand(&operand0);
model.addOperand(&operand1);
model.addOperand(&operand2);
model.addOperand(&operand3);
model.addOperand(&operand4);
model.addOperand(&operand5);
model.addOperation(&operation0);
model.addOperation(&operation1);
model.addOperation(&operation2);
model.addOperation(&operation3);
model.addOperation(&operation4);

T2C opt;
bool modified = false;
opt.run(&model, &modified);

model.echo();
return 0;
}
