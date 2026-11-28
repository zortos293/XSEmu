# Fetches NVIDIA's Streamline runtime and reports what a DLSS build needs.
#
# NVIDIA ships the SDK behind a developer-programme login, so this script downloads nothing on its
# own: it unpacks an SDK zip you downloaded, copies the four runtime DLLs the emulator loads into
# the folder xcl looks at, and says exactly what is still missing. No NVIDIA binary is committed
# to the tree and the target folder is git-ignored.
#
#   .\scripts\fetch-dlss.ps1 -SdkZip "$env:USERPROFILE\Downloads\streamline-sdk-2.14.16.zip"
#   .\scripts\fetch-dlss.ps1 -SdkZip ... -Build                 # also build xsemu-dlss-shim.dll
#   .\scripts\fetch-dlss.ps1 -Check                             # only say what is missing
#
# The runtime source can be either an SDK zip (lib/x64 + include) or an installed game folder that
# already has the DLLs; -From copies from such a folder.

[CmdletBinding()]
param(
    [string]$SdkZip,
    [string]$From,
    [string]$Into = (Join-Path $PSScriptRoot '..\dlss'),
    [switch]$Build,
    [switch]$Check
)

$ErrorActionPreference = 'Stop'
$wanted = @('sl.interposer.dll', 'sl.common.dll', 'sl.dlss.dll', 'nvngx_dlss.dll')

function Say($m) { Write-Host "fetch-dlss: $m" }

$Into = [IO.Path]::GetFullPath($Into)
Say "target folder $Into"

if ($Check) {
    $missing = $wanted | Where-Object { -not (Test-Path (Join-Path $Into $_)) }
    $shim = Join-Path $Into 'xsemu-dlss-shim.dll'
    if ($missing.Count -eq 0) { Say "all four runtime DLLs present" } else { Say ("missing: " + ($missing -join ', ')) }
    if (Test-Path $shim) { Say "shim present: $shim" } else { Say "shim missing: $shim (build it or use -Build)" }
    exit 0
}

if (-not (Test-Path $Into)) { New-Item -ItemType Directory -Path $Into | Out-Null }

if ($From) {
    if (-not (Test-Path $From)) { throw "no such folder: $From" }
    foreach ($dll in $wanted) {
        $p = Join-Path $From $dll
        if (Test-Path $p) { Copy-Item -LiteralPath $p -Destination $Into -Force; Say "copied $dll" }
        else { Say "not in ${From}: $dll" }
    }
}

if ($SdkZip) {
    if (-not (Test-Path $SdkZip)) { throw "no such zip: $SdkZip" }
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("streamline-" + [Guid]::NewGuid().ToString('N').Substring(0, 8))
    New-Item -ItemType Directory -Path $tmp | Out-Null
    try {
        Say "unpacking $SdkZip"
        Expand-Archive -LiteralPath $SdkZip -DestinationPath $tmp -Force
        # SDK zips nest the root folder differently per release: find the DLLs rather than assume a path.
        $found = Get-ChildItem -Path $tmp -Recurse -Include $wanted -File | Select-Object -First 20
        if (-not $found) { throw "no Streamline DLLs inside $SdkZip (is this the SDK zip, not a runtime installer?)" }
        foreach ($dll in $wanted) {
            $src = $found | Where-Object { $_.Name -eq $dll } | Select-Object -First 1
            if ($src) { Copy-Item -LiteralPath $src.FullName -Destination $Into -Force; Say "copied $dll from $($src.DirectoryName)" }
            else { Say "not in the zip: $dll" }
        }
        $include = Get-ChildItem -Path $tmp -Recurse -Filter 'sl_dlss.h' -File | Select-Object -First 1
        if ($include) {
            $sdkRoot = Split-Path (Split-Path $include.DirectoryName -Parent) -Parent
            Say "SDK headers at $sdkRoot; build with -Build or -DSTREAMLINE_DIR=$sdkRoot"
            if ($Build) {
                $cmake = Get-Command cmake -ErrorAction SilentlyContinue
                if (-not $cmake) { throw "-Build needs cmake on PATH (Visual Studio's C++ workload installs it)" }
                $src = Join-Path $PSScriptRoot '..\crates\xcl-dlss-shim'
                $bld = Join-Path $src 'build'
                & cmake -S $src -B $bld -A x64 -DSTREAMLINE_DIR=$sdkRoot
                if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }
                & cmake --build $bld --config Release
                if ($LASTEXITCODE -ne 0) { throw "shim build failed ($LASTEXITCODE)" }
                $built = Get-ChildItem -Path $bld -Recurse -Filter 'xsemu-dlss-shim.dll' -File | Select-Object -First 1
                if (-not $built) { throw "the build produced no xsemu-dlss-shim.dll" }
                Copy-Item -LiteralPath $built.FullName -Destination $Into -Force
                Say "built and copied xsemu-dlss-shim.dll"
            }
        }
        else { Say "no sl_dlss.h in the zip: the SDK headers are missing, so the shim cannot be built from it" }
    }
    finally { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
}

if (-not $From -and -not $SdkZip) {
    Say "nothing to copy: pass -SdkZip <zip you downloaded from developer.nvidia.com> or -From <folder with the DLLs>"
}

$missing = $wanted | Where-Object { -not (Test-Path (Join-Path $Into $_)) }
if ($missing.Count -eq 0) { Say "runtime complete: $Into" } else { Say ("still missing: " + ($missing -join ', ')) }
