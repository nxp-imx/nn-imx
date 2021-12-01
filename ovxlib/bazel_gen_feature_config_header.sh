#!/bin/sh

#generate pre-build header to config features
#echo Auto generate feature config header...
FEATURE_CONFIG_FILE=$1
echo "/****************************************************************************"
echo "*"
echo "*    Copyright (c) 2019 Vivante Corporation"
echo "*"
echo "*    Permission is hereby granted, free of charge, to any person obtaining a"
echo "*    copy of this software and associated documentation files (the "Software"),"
echo "*    to deal in the Software without restriction, including without limitation"
echo "*    the rights to use, copy, modify, merge, publish, distribute, sublicense,"
echo "*    and/or sell copies of the Software, and to permit persons to whom the"
echo "*    Software is furnished to do so, subject to the following conditions:"
echo "*"
echo "*    The above copyright notice and this permission notice shall be included in"
echo "*    all copies or substantial portions of the Software."
echo "*"
echo "*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR"
echo "*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,"
echo "*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE"
echo "*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER"
echo "*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING"
echo "*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER"
echo "*    DEALINGS IN THE SOFTWARE."
echo "*"
echo "*****************************************************************************/"
echo /*****Auto generated header file, Please DO NOT modify manually!*****/
echo "#ifndef _VSI_NN_FEATURE_CONFIG_H"
echo "#define _VSI_NN_FEATURE_CONFIG_H"
echo ""
IFS_old=$IFS
IFS=$'\n'
for line in `cat $FEATURE_CONFIG_FILE`
do
    echo "$line"
done
IFS=$IFS_old
echo ""
echo "#endif"
#Generate feature config header to bazel  successfully."
