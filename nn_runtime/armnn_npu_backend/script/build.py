#!/usr/bin/env python3
# Copyright (c) Vivante Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import sys
import os
import logging
import json
import subprocess
import multiprocessing

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.dirname(os.path.abspath(__file__))

class Env():
    def __init__(self, args):
        self.board_name = args.board_name
        self.args = args
        self.build = "Debug"
        self.driver_sdk_dir = args.driver_sdk_dir
        self.ovxlib_dir = args.ovxlib_dir
        self.blender_dir = args.blender_dir
        if args.cmake_path:
            self.cmake_path = args.cmake_path
        else:
            self.cmake_path = "cmake"
        self.out_dir = root_dir + "/build/" + self.board_name
        self.toolchain_dir = root_dir + "/toolchains"
        self.patch_dir = root_dir + "/patch"
        self.boost_dir = root_dir + "/armnn-devenv/boost"
        self.boost_out_dir = self.out_dir + "/boost"
        self.flatbuffers_dir = root_dir + "/armnn-devenv/flatbuffers"
        self.flatbuffers_host_dir = self.out_dir + "/flatbuffers_host"
        self.flatbuffers_target_dir = self.out_dir + "/flatbuffers_target"
        self.tflite_schema_dir = self.out_dir + "/tflite_schema"
        self.protobuf_dir = root_dir + "/armnn-devenv/google/protobuf"
        self.protobuf_host_dir = self.out_dir + "/protobuf_host"
        self.protobuf_target_dir = self.out_dir + "/protobuf_target"
        self.tensorflow_dir = root_dir + "/armnn-devenv/google/tensorflow"
        self.tf_pb_dir = self.out_dir + "/tf_pb"
        self.caffe_dir = root_dir + "/armnn-devenv/caffe"
        self.caffe_pb_dir = self.caffe_dir + "/.build_release/src"
        self.armnn_dir = root_dir + "/armnn-devenv/armnn"
        self.armnn_out_dir = self.out_dir + "/armnn"
        self.package_dir = self.out_dir + "/package"

logging.basicConfig(format="%(asctime)s %(name)s [%(levelname)s] - %(message)s", level=logging.DEBUG)
log = logging.getLogger("Build")

def run_subprocess(args, cwd=None, capture=False, shell=False, env={}):
    log.debug("Running subprocess in '{0}'\n{1}".format(cwd or os.getcwd(), args))
    my_env = os.environ.copy()

    stdout, stderr = (subprocess.PIPE, subprocess.STDOUT) if capture else (None, None)
    my_env.update(env)

    completed_process = subprocess.run(args, cwd=cwd, check=True,
        stdout=stdout, stderr=stderr, env=my_env, shell=shell)
    log.debug("Subprocess completed. Return code=" + str(completed_process.returncode))
    return completed_process

def find_board(env):
    for o in env.data_toolchain['toolchain']:
        if o['name'] == env.board_name:
            return o

def build_prepare(env):
    log.debug("Package all binary files...")

    args = ["git reset --hard"]
    run_subprocess(args, cwd=env.armnn_dir, shell=True)

    args = ["cp -r " + root_dir + "/armnn-devenv/test/armnn/model " +
        env.armnn_dir + "/tests"]
    run_subprocess(args, cwd=env.armnn_dir, shell=True)

    args = ["patch -p1<" + env.patch_dir + "/0001-VSI_NPU-the-patch-for-armnn-v21.02.patch"]
    run_subprocess(args, cwd=env.armnn_dir, shell=True)

def build_boost(env):
    log.debug("Build boost library...")
    board = find_board(env)
    if not os.path.exists(env.boost_out_dir):
        os.mkdir(env.boost_out_dir)
    with open(env.boost_out_dir + "/user-config.jam", mode="w") as f:
        f.write("using gcc : arm : " + board["prefix"] + "-g++ ;")
    run_subprocess(["./bootstrap.sh", "--prefix=" + env.boost_out_dir + "/install"], cwd=env.boost_dir,
        env={"PATH": board["root"] + board["host"] + ":" + os.getenv("PATH", default="")})

    args = ["./b2", "install", "link=static", 'cxxflags=-fPIC']
    args.append('cxxflags=--sysroot=' + board["root"] + board["sysroot"])
    for i in board["include"]:
        args.append("cxxflags=-isystem")
        args.append("cxxflags="+ board["root"] + i)
    args.extend(["--with-filesystem", "--with-test", "--with-log", "--with-program_options",
        "--user-config=" + env.boost_out_dir + "/user-config.jam",
        "-j{0}".format(multiprocessing.cpu_count())])

    run_subprocess(args, cwd=env.boost_dir,
        env={"PATH": board["root"] + board["host"] + ":" + os.getenv("PATH", default="")})

def build_flatbuffers_host(env):
    log.debug("Build flatbuffers library for host...")
    if not os.path.exists(env.flatbuffers_host_dir):
        os.mkdir(env.flatbuffers_host_dir)
    args = [env.cmake_path, "-G", "Unix Makefiles", "-DCMAKE_BUILD_TYPE=Release",
        "-DFLATBUFFERS_BUILD_TESTS=0",
        "-DCMAKE_INSTALL_PREFIX:PATH="+ env.flatbuffers_host_dir + "/install",
        env.flatbuffers_dir]
    run_subprocess(args, cwd=env.flatbuffers_host_dir)

    args = ["make", "all", "install", "-j{0}".format(multiprocessing.cpu_count())]
    run_subprocess(args, cwd=env.flatbuffers_host_dir)

def build_flatbuffers_target(env):
    log.debug("Build flatbuffers library for target...")
    board = find_board(env)
    if not os.path.exists(env.flatbuffers_target_dir):
        os.mkdir(env.flatbuffers_target_dir)
    args = [env.cmake_path + ' -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release' +
        " -DFLATBUFFERS_BUILD_TESTS=0" +
        ' -DCMAKE_CXX_FLAGS="-fPIC"' +
        " -DCMAKE_TOOLCHAIN_FILE=" + env.toolchain_dir + "/" + board['cmake'] +
        " -DCMAKE_INSTALL_PREFIX:PATH="+ env.flatbuffers_target_dir + "/install " +
        env.flatbuffers_dir]
    run_subprocess(args, cwd=env.flatbuffers_target_dir, shell=True)

    args = ["make", "all", "install", "-j{0}".format(multiprocessing.cpu_count())]
    run_subprocess(args, cwd=env.flatbuffers_target_dir)

def build_tflite_schema(env):
    log.debug("Generate TF Lite schema file...")
    if not os.path.exists(env.tflite_schema_dir):
        os.mkdir(env.tflite_schema_dir)
    args = ["cp", env.tensorflow_dir + "/tensorflow/lite/schema/schema.fbs", "."]
    run_subprocess(args, cwd=env.tflite_schema_dir)

    args = [env.flatbuffers_host_dir + "/install/bin/flatc", "-c", "--gen-object-api",
        "--reflect-types", "--reflect-names", "schema.fbs"]
    run_subprocess(args, cwd=env.tflite_schema_dir)

def build_protobuf_host(env):
    log.debug("Build protobuf library for host...")
    if not os.path.exists(env.protobuf_host_dir):
        os.mkdir(env.protobuf_host_dir)
    args = ["./autogen.sh"]
    run_subprocess(args, cwd=env.protobuf_dir)

    args = [env.protobuf_dir + "/configure", "--prefix=" + env.protobuf_host_dir + "/x86_pb_install"]
    run_subprocess(args, cwd=env.protobuf_host_dir)

    args = ["make", "install", "-j{0}".format(multiprocessing.cpu_count())]
    run_subprocess(args, cwd=env.protobuf_host_dir)

def build_protobuf_target(env):
    log.debug("Build protobuf library for target...")
    board = find_board(env)
    if not os.path.exists(env.protobuf_target_dir):
        os.mkdir(env.protobuf_target_dir)
    #args = ["./autogen.sh"]
    #run_subprocess(args, cwd=env.protobuf_dir)

    flags = "-fPIE -fPIC --sysroot=" + board["root"] + board["sysroot"] + " -D__arm64"

    for i in board["include"]:
        flags = flags + " -isystem " + board["root"] + i

    my_env = {
        "PATH": board["root"] + board["host"] + ":" + os.getenv("PATH", default=""),
        "CC": board["prefix"]+ "-gcc",
        "CXX": board["prefix"]+ "-g++",
        "CFLAGS": flags,
        "CXXFLAGS": flags,
        "CORSS_COMPILE": board["root"] + board["host"] + "/" + board["prefix"] + "-",
    }
    args = [env.protobuf_dir + "/configure", "--host=" + board["prefix"],
        "--prefix=" + env.protobuf_target_dir + "/arm64_pb_install",
        "--with-protoc=" + env.protobuf_host_dir + "/x86_pb_install/bin/protoc"]
    run_subprocess(args, cwd=env.protobuf_target_dir, env=my_env)

    args = ["make", "install", "-j{0}".format(multiprocessing.cpu_count())]
    run_subprocess(args, cwd=env.protobuf_target_dir, env=my_env)

def build_tf_pb(env):
    log.debug("Generate TensorFlow protobuf definitions...")
    args = [env.armnn_dir + "/scripts/generate_tensorflow_protobuf.sh", env.tf_pb_dir,
        env.protobuf_host_dir + "/x86_pb_install"]
    run_subprocess(args, cwd=env.tensorflow_dir)

def build_caffe_pb(env):
    log.debug("Generate Caffe protobuf definitions...")

    args = ["git reset --hard"]
    run_subprocess(args, cwd=env.caffe_dir, shell=True)

    with open(env.caffe_dir + "/Makefile.config.example", mode='r', newline='\n', encoding='UTF-8') as f0:
        lines = f0.readlines()

    replace_strs = [
        {
            "origin": '# CPU_ONLY := 1',
            "replace": 'CPU_ONLY := 1',
        },
        {
            "origin": '# USE_OPENCV := 0',
            "replace": 'USE_OPENCV := 0',
        },
        {
            "origin": 'INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include',
            "replace": 'INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/ ' +
                env.protobuf_host_dir + "/x86_pb_install/include/",
        },
        {
            "origin": 'LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib',
            "replace": 'LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib ' +
                '/usr/lib/x86_64-linux-gnu/hdf5/serial/ ' +
                env.protobuf_host_dir + "/x86_pb_install/lib/",
        },
    ]

    with open(env.caffe_dir + "/Makefile.config", mode='w', newline='\n', encoding='UTF-8') as f1:
        for index, line in enumerate(lines):
            for replace_str in replace_strs:
                if line.find(replace_str['origin']) != -1:
                    lines[index] = lines[index].replace(replace_str['origin'], replace_str['replace'])
        f1.writelines(lines)

    args = ["patch -p1<" + env.patch_dir + "/0002-VSI_NPU-the-patch-for-caffe.patch"]
    run_subprocess(args, cwd=env.caffe_dir, shell=True)

    my_env = {
        "PATH": env.protobuf_host_dir + "/x86_pb_install/bin/:" + os.getenv("PATH", default=""),
        "LD_LIBRARY_PATH": env.protobuf_host_dir + "/x86_pb_install/lib/:" +
            os.getenv("LD_LIBRARY_PATH", default=""),
    }
    args = ["make", "all", "-j{0}".format(multiprocessing.cpu_count())]
    run_subprocess(args, cwd=env.caffe_dir, env=my_env)

def build_armnn(env):
    log.debug("Build ArmNN library...")
    board = find_board(env)
    if not os.path.exists(env.armnn_out_dir):
        os.mkdir(env.armnn_out_dir)

    ld_flags = "-lpthread -L" + board["root"] + board["sysroot"] + "/lib -Wl,-rpath-link," \
        + env.driver_sdk_dir + "/drivers"
    my_env = {
        "VIVANTE_SDK_DIR": env.driver_sdk_dir,
        "OVXLIB_DIR": env.ovxlib_dir,
        "NNRT_ROOT": env.blender_dir,
        "LDFLAGS": ld_flags,
    }
    args = [env.cmake_path, env.armnn_dir, "-DBUILD_TF_PARSER=1", "-DARMNNREF=1",
        "-DCMAKE_BUILD_TYPE=" + env.args.config,
        "-DBUILD_GATORD_MOCK=0",
        "-DBoost_NO_SYSTEM_PATHS=1",
        "-DBoost_NO_BOOST_CMAKE=1",
        "-DBUILD_TF_PARSER=1",
        "-DBUILD_TF_LITE_PARSER=1",
        "-DBUILD_CAFFE_PARSER=1",
        "-DVSI_NPU=1",
        "-DBUILD_VSI_TESTS=1",
        "-DARMNN_ROOT=" + env.armnn_dir,
        "-DBOOST_ROOT=" + env.boost_out_dir + "/install",
        "-DPROTOBUF_ROOT=" + env.protobuf_target_dir + "/arm64_pb_install",
        "-DTF_GENERATED_SOURCES=" + env.tf_pb_dir,
        "-DTF_LITE_SCHEMA_INCLUDE_PATH=" + env.tflite_schema_dir,
        "-DFLATBUFFERS_ROOT=" + env.flatbuffers_target_dir + "/install",
        "-DCAFFE_GENERATED_SOURCES=" + env.caffe_pb_dir,
        "-DCMAKE_TOOLCHAIN_FILE=" + env.toolchain_dir + "/" + board['cmake']]
    run_subprocess(args, cwd=env.armnn_out_dir, env=my_env)

    args = ["make", "-j{0}".format(multiprocessing.cpu_count())]
    run_subprocess(args, cwd=env.armnn_out_dir)

def build_package(env):
    log.debug("Package all binary files...")
    board = find_board(env)
    if not os.path.exists(env.package_dir):
        os.mkdir(env.package_dir)

    args = ["cp -d " + env.armnn_out_dir + "/*.so* ."]
    run_subprocess(args, cwd=env.package_dir, shell=True)

    args = ["cp -d " + env.armnn_out_dir + "/UnitTests ."]
    run_subprocess(args, cwd=env.package_dir, shell=True)

    args = ["cp -d " + env.armnn_out_dir + "/tests/model/Caffe* ."]
    run_subprocess(args, cwd=env.package_dir, shell=True)

    args = ["cp -d " + env.armnn_out_dir + "/tests/model/Tf* ."]
    run_subprocess(args, cwd=env.package_dir, shell=True)

    args = ["cp -d " + env.protobuf_target_dir + "/arm64_pb_install/lib/*.so* ."]
    run_subprocess(args, cwd=env.package_dir, shell=True)

def parse_arguments(board_list):
    parser = argparse.ArgumentParser(description="Vsi_npu CI build tool for Armnn.")
    parser.add_argument("board_name", type=str, choices=board_list, help="Board Name.")
    parser.add_argument("driver_sdk_dir", type=str, help="Driver SDK directory.")
    parser.add_argument("blender_dir", type=str, help="Blender root directory.")
    parser.add_argument("ovxlib_dir", type=str, help="Ovxlib directory.")
    parser.add_argument("--config", type=str, default="Debug",
                        choices=["Debug", "Release", "RelWithDebInfo"],
                        help="Configuration(s) to build.")
    parser.add_argument("--build_all", action='store_true', help="Build All for Armnn.")
    parser.add_argument("--build_prepare", action='store_true', help="Prepare build.")
    parser.add_argument("--build_boost", action='store_true', help="Build Boost library.")
    parser.add_argument("--build_flatbuffers_host", action='store_true', help="Build Flatbuffer library for host.")
    parser.add_argument("--build_flatbuffers_target", action='store_true',
                        help="Build Flatbuffer library for target.")
    parser.add_argument("--build_tflite_schema", action='store_true', help="Generate TF Lite schema file.")
    parser.add_argument("--build_protobuf_host", action='store_true', help="Build protobuf library for host.")
    parser.add_argument("--build_protobuf_target", action='store_true', help="Build protobuf library for target.")
    parser.add_argument("--build_tf_pb", action='store_true', help="Generate TensorFlow protobuf definitions.")
    parser.add_argument("--build_caffe_pb", action='store_true', help="Generate Caffe protobuf definitions.")
    parser.add_argument("--build_armnn", action='store_true', help="Build ArmNN library.")
    parser.add_argument("--build_package", action='store_true', help="Package all binary files.")
    parser.add_argument("--cmake_path", type=str, help="Cmake tool path.")
    return parser.parse_args()

def main():
    with open(script_dir + "/toolchain.json") as f:
        data_toolchain = json.load(f)

    board_list = []
    for o in data_toolchain['toolchain']:
        board_list.append(o["name"])

    args = parse_arguments(board_list)

    env = Env(args)
    env.data_toolchain = data_toolchain

    if not os.path.exists(env.out_dir):
        os.makedirs(env.out_dir)

    if args.build_all or args.build_prepare:
        build_prepare(env)

    if args.build_all or args.build_boost:
        build_boost(env)

    if args.build_all or args.build_flatbuffers_host:
        build_flatbuffers_host(env)

    if args.build_all or args.build_flatbuffers_target:
        build_flatbuffers_target(env)

    if args.build_all or args.build_tflite_schema:
        build_tflite_schema(env)

    if args.build_all or args.build_protobuf_host:
        build_protobuf_host(env)

    if args.build_all or args.build_protobuf_target:
        build_protobuf_target(env)

    if args.build_all or args.build_tf_pb:
        build_tf_pb(env)

    if args.build_all or args.build_caffe_pb:
        build_caffe_pb(env)

    if args.build_all or args.build_armnn:
        build_armnn(env)

    if args.build_all or args.build_package:
        build_package(env)

if __name__ == "__main__":
    sys.exit(main())
