@echo off
REM Foolproof launcher for the W: server (or any machine).
REM Uses the REAL Python 3.12 via the "py" launcher, NOT the "python" on PATH
REM (which may be Claude's bundled hermes-agent venv with no pip).
cd /d "%~dp0"
echo ==== updating from GitHub ====
git pull
echo.
echo ==== installing into Python 3.12 (editable, first run pulls deps) ====
py -3.12 -m pip install -e .
echo.
echo ==== version check (should say 0.7.4) ====
py -3.12 -m nl2tags version
echo.
echo ==== starting studio — open http://127.0.0.1:8000 in your browser ====
echo (leave this window open; close it to stop the server)
py -3.12 -m nl2tags studio
pause
