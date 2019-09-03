package(
    default_visibility = ["//visibility:public"],
    features = ["-parse_headers"],
)

cc_library(
    name = "ovxlibnn",
    copts = ["-Werror", "-fvisibility=hidden", '-DOVXLIB_API=__attribute__((visibility(\\"default\\")))'],
    includes = [
        ".",
    ],
    hdrs = [
        "NeuralNetworks.h",
        "model.h",
        "operation.h",
        "operand.h",
        "file_map_memory.h",
        "types.h",
        "event.h",
        "error.h",
        "utils.h",
        "execution.h",
        "compilation.h",
        "prepared_model.h",
        "ovxlib_delegate.h",
        "graph_transformations/transformations.h",
        "memory_pool.h",
    ],
    srcs = [
        "NeuralNetworks.cc",
        "model.cc",
        "operation.cc",
        "operand.cc",
        "file_map_memory.cc",
        "compilation.cc",
        "event.cc",
        "utils.cc",
        "execution.cc",
        "prepared_model.cc",
        "ovxlib_delegate.cc",
        "graph_transformations/t2c.cc",
        "graph_transformations/align_broadcast_op.cc",
        "graph_transformations/transformations.cc",
        "graph_transformations/optimize_permute.cc",
        "graph_transformations/fp32tofp16.cc",
        "graph_transformations/validate_quantized_graph.cc",
        "graph_transformations/nnapi_interpreter.cc",
        "memory_pool.cc"
    ],
    deps = ["//:ovxlib"]

)

cc_binary(
    name = "libneuralnetworks.so",
    copts = ["-Werror", "-fvisibility=hidden", '-DOVXLIB_API=__attribute__((visibility(\\"default\\")))'],
    deps = ["//nnapi:ovxlibnn"],
    linkshared=1,
    )


cc_binary(
    name = "t2c_test",
    copts = ["-Werror", "-fvisibility=hidden", '-DOVXLIB_API=__attribute__((visibility(\\"default\\")))'],
    includes = [
        ".",
    ],
    srcs = [
        "graph_transformations/t2c_test.cc",
    ],
    deps = ["ovxlibnn"]
)
