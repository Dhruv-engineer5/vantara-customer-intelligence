# Creates .env with a random local database password (Windows PowerShell). Run from the project folder:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# An existing .env is never overwritten. The password is not shown and not stored anywhere else.
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root ".env"
if (Test-Path $target) { Write-Host ".env already exists; leaving it unchanged."; exit 0 }
$chars = (48..57) + (65..90) + (97..122)
$pw = -join ($chars | Get-Random -Count 28 | ForEach-Object { [char]$_ })
(Get-Content (Join-Path $root ".env.example")) -replace "(?m)^POSTGRES_PASSWORD=.*$", "POSTGRES_PASSWORD=$pw" | Set-Content $target
Write-Host "Created .env with a random local database password."
