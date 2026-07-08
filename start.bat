@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
:menu
cls
echo(
echo    waifu-nl2tags
echo    ============================
echo    [1]  训练台  (studio, 8000)
echo    [2]  翻译页面 (serve, 8001)
echo    [3]  两个都开
echo    [Q]  退出
echo(
set /p c=选择: 
if /I "%c%"=="1" goto studio
if /I "%c%"=="2" goto serve
if /I "%c%"=="3" goto both
if /I "%c%"=="Q" exit /b
goto menu
:studio
start "nl2tags studio" cmd /k nl2tags studio --port 8000
timeout /t 3 >nul & start "" http://127.0.0.1:8000
goto done
:serve
if exist "out\adapter\adapter_config.json" ( start "nl2tags serve" cmd /k nl2tags serve --adapter out\adapter --port 8001 ) else ( start "nl2tags serve" cmd /k nl2tags serve --proxy --port 8001 )
timeout /t 3 >nul & start "" http://127.0.0.1:8001
goto done
:both
start "nl2tags studio" cmd /k nl2tags studio --port 8000
if exist "out\adapter\adapter_config.json" ( start "nl2tags serve" cmd /k nl2tags serve --adapter out\adapter --port 8001 ) else ( start "nl2tags serve" cmd /k nl2tags serve --proxy --port 8001 )
timeout /t 3 >nul & start "" http://127.0.0.1:8000
goto done
:done
echo(
echo   已启动。此窗口可关闭。
timeout /t 3 >nul
