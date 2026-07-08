# One-line install (Windows PowerShell):
#   irm https://raw.githubusercontent.com/USER/waifu-nl2tags/main/install.ps1 | iex
$ErrorActionPreference = "Stop"
$Repo = if ($env:REPO) { $env:REPO } else { "git+https://github.com/USER/waifu-nl2tags.git" }
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "waifu-nl2tags[train] @ $Repo"
Write-Host "`n"; nl2tags doctor
Write-Host "`nReady. Try:  nl2tags quickstart"
