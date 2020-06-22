package(default_visibility = ["//visibility:public"])

filegroup(
    name = "libs",
    srcs = glob([
        "lib/*.*",
        "drivers/*.*",
    ], exclude = [
        "lib/libovxnne.so",
        "lib/libvsivxquant.so",
        "lib/libshaderc_shared.so",
        "lib/libvsivxext.so",
    ]),
)

cc_library(
    name = "VIV_SDK_LIB",
    hdrs = glob([
        "include/**/*.h"
    ]),
    srcs = select({
        "//conditions:default": [":dev_libs"],
        ":development": [":dev_libs"],
        ":release": [":libs"],
    }),
)

cc_library(
    name = "VIV_SDK_INC",
    includes = [
        "include",
        "include/VX",
    ],
    hdrs = glob([
        "include/**/*.h"
    ]),
)

filegroup(
    name = "VIV_EXT_HDR",
    srcs = [
        "include/CL/cl_viv_vx_ext.h",
    ],
)

config_setting(
    name = "release",
    define_values = {
        "mode": "rel",
    },
)

config_setting(
    name = "development",
    define_values = {
        "mode": "dev",
    },
)

config_setting(
    name = "i386_linux",
    define_values = {
        "platform": "i386_linux",
    },
)

config_setting(
    name = "win32",
    define_values = {
        "platform": "win32",
    },
)

config_setting(
    name = "x64",
    define_values = {
        "platform": "x64",
    },
)

config_setting(
    name = "x64_linux",
    define_values = {
        "platform": "x64_linux",
    },
)

filegroup(
    name = "dev_libs",
    srcs = select({
        ":i386_linux": glob([
            "lib/i386_linux/*.*",
        ]),
        ":win32": glob([
            "lib/win32/*.so",
        ]),
        ":x64": glob([
            "lib/x64/*.so",
        ]),
        ":x64_linux": glob([
            "lib/x64_linux/*.*",
        ]),
        "//conditions:default": glob([
            "lib/x64_linux/*.*",
        ]),
    }),
)
