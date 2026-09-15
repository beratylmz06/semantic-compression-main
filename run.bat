@echo off
setlocal EnableExtensions
title Signal Compression Lab
pushd "%~dp0"

echo ==========================================================
echo   Signal Compression Lab
echo   DCT . Wavelet . Rate-Distortion Analysis
echo ==========================================================
echo.

rem --- find a real Python 3 (reject the Windows Store stub) -----------------
set "PY="
python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY py -3 -c "import sys" >nul 2>nul && set "PY=py -3"
if not defined PY goto :no_python

for /f "delims=" %%V in ('%PY% --version 2^>^&1') do set "PYVER=%%V"
echo   Python : %PY%   %PYVER%
echo   Folder : %CD%
echo.

rem --- dependencies -------------------------------------------------------------
%PY% -c "import gradio,numpy,cv2,scipy,pywt,skimage,PIL,matplotlib,plotly" >nul 2>nul
if errorlevel 1 goto :install
goto :after_install

:install
echo First run: installing requirements - this can take a few minutes...
echo.
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto :install_failed
echo.

:after_install
rem --- free port 7860 if a previous run is still holding it -----------------
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Write-Host ('  Freeing port 7860 - stopping previous instance, PID ' + $_); Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }" 2>nul

echo.
echo   Starting on  http://127.0.0.1:7860
echo   The browser opens automatically. Keep this window open;
echo   press Ctrl+C here or close the window to stop the app.
echo ----------------------------------------------------------
echo.
%PY% app.py
set "RC=%errorlevel%"
echo.
echo App stopped. Exit code %RC%.
goto :end

:no_python
echo [ERROR] Python 3 was not found on this PC.
echo         1. Install it from  https://www.python.org/downloads/
echo         2. During setup, tick  "Add python.exe to PATH"
echo         3. Reopen this file.
goto :end

:install_failed
echo.
echo [ERROR] Installing the requirements failed - read the messages above.
goto :end

:end
echo.
pause
popd
endlocal
