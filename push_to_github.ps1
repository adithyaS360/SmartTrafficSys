# Commit and push SmartTrafficSys to GitHub.
#
#   Right-click this file -> "Run with PowerShell"
#   or:  powershell -ExecutionPolicy Bypass -File push_to_github.ps1
#
# It shows you exactly what will be committed and waits for you to confirm
# before doing anything. Nothing is deleted and nothing is force-pushed.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Step($text) { Write-Host "`n>> $text" -ForegroundColor Cyan }
function Warn($text) { Write-Host "   $text" -ForegroundColor Yellow }
function Bad($text)  { Write-Host "   $text" -ForegroundColor Red }

Write-Host "==================================================================="
Write-Host "  SmartTrafficSys - commit and push"
Write-Host "==================================================================="

# --- sanity checks --------------------------------------------------------

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Bad "git is not installed, or not on PATH."
    Warn "Install it from https://git-scm.com/download/win and run this again."
    Read-Host "`nPress Enter to close"; exit 1
}

if (-not (Test-Path ".git")) {
    Bad "This folder is not a git repository."
    Warn "Expected to find .git in $PSScriptRoot"
    Read-Host "`nPress Enter to close"; exit 1
}

$remote = git remote get-url origin 2>$null
if (-not $remote) {
    Bad "No 'origin' remote is configured."
    Warn "Add one with:  git remote add origin <your repo url>"
    Read-Host "`nPress Enter to close"; exit 1
}
Write-Host "`n   repository : $PSScriptRoot"
Write-Host "   remote     : $remote"
Write-Host "   branch     : $(git rev-parse --abbrev-ref HEAD)"

# --- identity -------------------------------------------------------------
# Only set it if it is missing, so an existing global config is never clobbered.

Step "Checking commit identity"
$name  = git config user.name
$email = git config user.email
if (-not $name -or -not $email) {
    Warn "No commit identity set for this repository."
    $name  = Read-Host "   Your name for commits"
    $email = Read-Host "   Your email for commits"
    git config user.name  "$name"
    git config user.email "$email"
}
Write-Host "   committing as: $name <$email>"

# --- untrack the old scaffold --------------------------------------------
# It stays on disk; this only stops git tracking it. Nothing is deleted.

Step "Untracking the superseded scaffold folder"
$tracked = git ls-files "Smart_Traffic_Management_System" 2>$null
if ($tracked) {
    git rm -r --cached "Smart_Traffic_Management_System" --quiet
    Write-Host "   removed from the index (the folder is still on disk)"
} else {
    Write-Host "   not tracked - nothing to do"
}

# --- stage ----------------------------------------------------------------

Step "Staging changes"
git add -A

# Split added/modified from deleted. This distinction matters: the safety
# check below must only look at files ENTERING the tree. Using the unfiltered
# list would flag the scaffold's own removal as a forbidden file and abort the
# script on the very step that is cleaning the repository up.
$entering = @(git diff --cached --name-only --diff-filter=d)
$leaving  = @(git diff --cached --name-only --diff-filter=D)

if (-not $entering -and -not $leaving) {
    Warn "Nothing to commit - the working tree matches the last commit."
    Read-Host "`nPress Enter to close"; exit 0
}

if ($entering) {
    Write-Host "`n   Files to be committed ($($entering.Count)):" -ForegroundColor Green
    $entering | ForEach-Object { Write-Host "     $_" }
}
if ($leaving) {
    Write-Host "`n   Removed from tracking ($($leaving.Count)) - still on disk:" -ForegroundColor Yellow
    $leaving | Select-Object -First 8 | ForEach-Object { Write-Host "     $_" }
    if ($leaving.Count -gt 8) { Write-Host "     ... and $($leaving.Count - 8) more" }
}

# --- safety check ---------------------------------------------------------
# Large generated files must never enter history: git keeps every version
# forever, so deleting them in a later commit does not shrink the repository.

Step "Checking for files that should not be committed"
$forbidden = $entering | Where-Object {
    $_ -match '\.(db|sqlite3?|pt|pth|onnx|weights|keras|h5|npz|mp4|avi|mov|mkv)$' -or
    $_ -eq '.env' -or $_ -match 'Smart_Traffic_Management_System/'
}
if ($forbidden) {
    Bad "These should NOT be committed:"
    $forbidden | ForEach-Object { Bad "     $_" }
    Warn "The .gitignore rules are not matching. Stopping so nothing large or"
    Warn "secret enters the history. Unstage with:  git reset"
    Read-Host "`nPress Enter to close"; exit 1
}
$bytes = ($entering | Where-Object { Test-Path $_ } |
          ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum
Write-Host ("   clean - nothing large or secret staged ({0:N0} KB total)" -f ($bytes / 1KB))

# --- confirm --------------------------------------------------------------

Write-Host ""
$answer = Read-Host "Commit and push? (y/N)"
if ($answer -notmatch '^[Yy]') {
    Warn "Cancelled. Your changes are still staged; run 'git reset' to unstage."
    Read-Host "`nPress Enter to close"; exit 0
}

# --- commit ---------------------------------------------------------------

Step "Committing"
$subject = "Rebuild: adaptive traffic control with CV detection and forecasting"
$body = @"
Complete rewrite of the scaffold into a working system.

- YOLOv8 detection with centroid tracking and line counting, making flow
  rate, queue length and dwell time measurable
- Signal controller as a state machine with hard safety invariants and
  pluggable strategies (fixed, adaptive, ML)
- Closed-loop simulator; adaptive control reduces average delay 29.6%
  against the best fixed-time schedule
- LSTM flow forecasting, +27.5% skill over a seasonal-naive baseline
- SQLAlchemy persistence with Alembic migrations, SQLite or PostgreSQL
- Flask REST API and live dashboard
- 174 tests across five suites
"@
git commit -m $subject -m $body
if ($LASTEXITCODE -ne 0) {
    Bad "The commit failed - see the message above."
    Read-Host "`nPress Enter to close"; exit 1
}

# --- push -----------------------------------------------------------------

Step "Pushing to origin"
$branch = git rev-parse --abbrev-ref HEAD
git push -u origin $branch
if ($LASTEXITCODE -ne 0) {
    Bad "The push failed."
    Warn "Common causes:"
    Warn "  - not signed in: run 'git push' once manually to authenticate"
    Warn "  - the remote has commits you do not: run 'git pull --rebase' first"
    Warn "Nothing was lost - your commit is saved locally either way."
    Read-Host "`nPress Enter to close"; exit 1
}

Write-Host "`n===================================================================" -ForegroundColor Green
Write-Host "  Pushed '$branch' to $remote" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green
Write-Host @"

  Your local branch is '$branch'. If the repository's default branch on
  GitHub is still 'master', change it under Settings -> Branches so people
  landing on the repo see this code.
"@
Read-Host "`nPress Enter to close"
