param(
    [string]$Remote = "lonestar6",
    [string]$SSHMac = "",
    [switch]$Send
)
$ErrorActionPreference = "Stop"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$stage = Join-Path $repo "src_3D/build/transfer"
New-Item -ItemType Directory -Force $stage | Out-Null
$archive = Join-Path $stage "fluid3d64.tar.gz"
& tar -czf $archive -C $repo --exclude=src_3D/build --exclude=src_3D/.gitignore src_3D
if ($LASTEXITCODE -ne 0) { throw "Source packaging failed" }
Write-Host "Archive ready at $archive"
if ($Send) {
    $scpOptions = @()
    if ($SSHMac) { $scpOptions = @("-o", "MACs=$SSHMac") }
    & scp @scpOptions $archive "${Remote}:~/fluid3d64.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "Transfer failed" }
    Write-Host "Uploaded to ~/fluid3d64.tar.gz on $Remote"
}
