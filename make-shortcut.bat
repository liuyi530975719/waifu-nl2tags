@echo off
setlocal
set "TARGET=%~dp0start.bat"
if not exist "%TARGET%" ( echo start.bat not found next to this script & pause & exit /b 1 )
powershell -NoProfile -ExecutionPolicy Bypass -Command "$t=$env:TARGET;$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\waifu-nl2tags.lnk');$s.TargetPath=$t;$s.WorkingDirectory=(Split-Path $t);$s.IconLocation='shell32.dll,220';$s.Save()"
echo.
echo 已在桌面创建 waifu-nl2tags 图标 -^> %TARGET%
pause
