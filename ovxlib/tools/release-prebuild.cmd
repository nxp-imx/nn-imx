SET SolutionDir=%1
SET ProjectDir=%2

SET SolutionDir=%SolutionDir:"=%
SET ProjectDir=%ProjectDir:"=%
SET DEST_FILE=%SolutionDir%ovxlib.path.user
SET TMP_FILE=%SolutionDir%ovxlib.path.user.tmp

ECHO %ProjectDir%> "%TMP_FILE%"

IF EXIST %DEST_FILE% (
    FC "%TMP_FILE%" "%DEST_FILE%" > nul 2>&1 || MOVE /Y "%TMP_FILE%" "%DEST_FILE%"
    DEL /F /Q "%TMP_FILE%" > nul 2>&1
) ELSE (
    MOVE /Y "%TMP_FILE%" "%DEST_FILE%"
)
SET DEST_FILE=%ProjectDir%include\vsi_nn_feature_config.h
SET TMP_FILE=%ProjectDir%include\vsi_nn_feature_config.h.tmp
SET FEATURE_CONFIG_SRC=%ProjectDir%vsi_feature_config
IF "%FEATURE_CONFIG_SRC%" NEQ "%FEATURE_CONFIG_SRC: =%" (
    ECHO "ERROR: The path of '%FEATURE_CONFIG_SRC%' contains spaces"
    EXIT /b -2
)

ECHO Auto generate feature config header...
ECHO /****************************************************************************> "%TMP_FILE%"
ECHO *>> "%TMP_FILE%"
ECHO *    Copyright (c) 2020 Vivante Corporation>> "%TMP_FILE%"
ECHO *>> "%TMP_FILE%"
ECHO *    Permission is hereby granted, free of charge, to any person obtaining a>> "%TMP_FILE%"
ECHO *    copy of this software and associated documentation files (the "Software"),>> "%TMP_FILE%"
ECHO *    to deal in the Software without restriction, including without limitation>> "%TMP_FILE%"
ECHO *    the rights to use, copy, modify, merge, publish, distribute, sublicense,>> "%TMP_FILE%"
ECHO *    and/or sell copies of the Software, and to permit persons to whom the>> "%TMP_FILE%"
ECHO *    Software is furnished to do so, subject to the following conditions:>> "%TMP_FILE%"
ECHO *>> "%TMP_FILE%"
ECHO *    The above copyright notice and this permission notice shall be included in>> "%TMP_FILE%"
ECHO *    all copies or substantial portions of the Software.>> "%TMP_FILE%"
ECHO *>> "%TMP_FILE%"
ECHO *    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR>> "%TMP_FILE%"
ECHO *    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,>> "%TMP_FILE%"
ECHO *    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE>> "%TMP_FILE%"
ECHO *    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER>> "%TMP_FILE%"
ECHO *    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING>> "%TMP_FILE%"
ECHO *    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER>> "%TMP_FILE%"
ECHO *    DEALINGS IN THE SOFTWARE.>> "%TMP_FILE%"
ECHO *>> "%TMP_FILE%"
ECHO *****************************************************************************/>> "%TMP_FILE%"
ECHO /*****Auto generated header file, Please DO NOT modify manually!*****/>> "%TMP_FILE%"
ECHO #ifndef _VSI_NN_FEATURE_CONFIG_H>> "%TMP_FILE%"
ECHO #define _VSI_NN_FEATURE_CONFIG_H>> "%TMP_FILE%"
ECHO.>> "%TMP_FILE%"
FOR /f "delims=" %%i in (%FEATURE_CONFIG_SRC%) do (
    ECHO %%i>> "%TMP_FILE%"
)
ECHO.>> "%TMP_FILE%"
ECHO #endif>> "%TMP_FILE%"

IF EXIST %DEST_FILE% (
    FC "%TMP_FILE%" "%DEST_FILE%" > nul 2>&1 || MOVE /Y "%TMP_FILE%" "%DEST_FILE%"
    DEL /F /Q "%TMP_FILE%" > nul 2>&1
) ELSE (
    MOVE /Y "%TMP_FILE%" "%DEST_FILE%"
)
ECHO Generate feature config header to %DEST_FILE% successfully.
