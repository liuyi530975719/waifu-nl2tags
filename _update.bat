@echo off
setlocal
cd /d E:\waifumaster\waifu-nl2tags
set LOG=%CD%\_update.log
echo === update start === > "%LOG%"
git add -A >> "%LOG%" 2>&1
git commit -m "civitai+grok data collector + studio Civitai step & keys panel (keys never committed); v0.4.0" >> "%LOG%" 2>&1
git push >> "%LOG%" 2>&1
echo EXITCODE=%ERRORLEVEL% >> "%LOG%"
echo === done === >> "%LOG%"
