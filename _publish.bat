@echo off
setlocal
set REPO=E:\waifumaster\waifu-nl2tags
set URL=https://github.com/liuyi530975719/waifu-nl2tags.git
cd /d "%REPO%"
set LOG=%REPO%\_publish.log
echo === publish start === > "%LOG%"
git --version >> "%LOG%" 2>&1
gh --version >> "%LOG%" 2>&1
if exist .git rmdir /s /q .git
git init >> "%LOG%" 2>&1
git branch -M main >> "%LOG%" 2>&1
git add -A >> "%LOG%" 2>&1
git commit -m "init: waifu-nl2tags" >> "%LOG%" 2>&1
echo --- try gh create+push --- >> "%LOG%"
gh repo create liuyi530975719/waifu-nl2tags --public --source=. --remote=origin --push >> "%LOG%" 2>&1
if errorlevel 1 (
  echo --- gh unavailable/exists, fallback to git push --- >> "%LOG%"
  git remote remove origin >> "%LOG%" 2>&1
  git remote add origin "%URL%" >> "%LOG%" 2>&1
  git push -u origin main >> "%LOG%" 2>&1
)
echo EXITCODE=%ERRORLEVEL% >> "%LOG%"
gh repo view liuyi530975719/waifu-nl2tags --json url --jq .url >> "%LOG%" 2>&1
echo === done === >> "%LOG%"
