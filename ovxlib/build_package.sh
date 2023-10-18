#!/bin/bash

BUILDTYPE="dev"
DELIVERABLE=./ovxlib-package-$BUILDTYPE

# Package final deliverable
package_deliverable()
{
    # Wipe existing devlierable
    echo "clean deliverable ..."
    rm -rf $DELIVERABLE;

    echo "packaging deliverable ..."

    # Copy souce code of ovxlib
    if [ "$BUILDTYPE" = "dev" ] || [ "$BUILDTYPE" = "rel" ]; then
        mkdir $DELIVERABLE
        for item in ./*
            do
                if [ "$item" != "$DELIVERABLE" ]; then
                    echo "copy $item to $DELIVERABLE"
                    cp $item $DELIVERABLE -rf
                else
                    echo "Ignore $item"
                fi
            done
    else
        exit $?
    fi

    # remove redundant folders and files
    rm -rf $DELIVERABLE/src/lcov
    rm -rf $DELIVERABLE/include/lcov
    rm -f $DELIVERABLE/ovxlib.vcxproj
    rm -f $DELIVERABLE/ovxlib.vcxproj.filters
    rm -f $DELIVERABLE/ovxlib.vcxproj.user
    rm -rf $DELIVERABLE/.git
    rm -rf $DELIVERABLE/test/internal
    rm -rf $DELIVERABLE/third-party/gperftools
    rm -rf $DELIVERABLE/tools/rockchips_toybrick_compiler

    # Tar
    PKG_FILE=$DELIVERABLE'-'`date +%m%d%Y`.tar.gz
    rm -rf $PKG_FILE
    tar -zcf $PKG_FILE $DELIVERABLE
    echo "Build package completed!"
}

if [ $# -ne 0 ]; then
    BUILDTYPE=$1
fi

if [ "$BUILDTYPE" = "dev" ]; then
    echo "Development Source Package Build Starting ..."
else
    echo "./build_package.sh [dev]"
    exit $?
fi


# Package
package_deliverable

