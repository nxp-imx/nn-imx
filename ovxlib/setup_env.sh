# optional arguments: SDK_PATH CONFIG
CONFIG="VIP8000_NANOD"

if [ "x$1" != "x" ]; then
    echo "Using local SDK at $1"
    SDK_PATH=$1
    export VIV_SDK_DIR=$SDK_PATH
    if [[ $SDK_PATH == *"cmdtools"* ]]; then
        SDK_TYPE=rel
    else
        SDK_TYPE=dev
    fi
else
    echo "No local SDK specified, using http archive specified in WORKSPACE"
    SDK_PATH=`pwd`/bazel-$(basename $PWD)/external/VIV_SDK
    unset VIV_SDK_DIR
    SDK_TYPE=rel
fi

if [ "x$2" != "x" ]; then
    CONFIG=$2
fi

# Do Not Modify
export VIVANTE_SDK_DIR=$SDK_PATH
export VSIMULATOR_CONFIG=$CONFIG

echo "Setup .bazelrc"
echo "test --test_env VIVANTE_SDK_DIR=\"$SDK_PATH\"" > .bazelrc
echo "test --test_env VSIMULATOR_CONFIG=$CONFIG" >> .bazelrc
echo "test --test_timeout=3600" >> .bazelrc
echo "build --define mode=$SDK_TYPE" >> .bazelrc

echo "Done"
