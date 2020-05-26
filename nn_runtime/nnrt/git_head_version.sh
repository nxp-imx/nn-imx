#!/bin/sh
HEAD_VERSION=`git log -n 1 --format=%h`

if [ $? -eq 0 ]
then
    echo $HEAD_VERSION
fi
