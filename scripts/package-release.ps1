param(
    [string] $ReleaseTag,
    [string] $BuildRoot
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $ReleaseTag) {
    $ReleaseTag = 'v' + (Get-Content -LiteralPath (Join-Path $RepoRoot 'VERSION') -Raw).Trim()
}
if (-not $BuildRoot) {
    $BuildRoot = Join-Path $RepoRoot 'build-windows'
}
$DistRoot = Join-Path $RepoRoot 'dist'
$ProductName = "ninfer-rtx4090-windows-x64-$ReleaseTag"
$ProductRoot = Join-Path $DistRoot $ProductName
$ArchivePath = Join-Path $DistRoot "$ProductName.zip"
$ChecksumPath = Join-Path $DistRoot "SHA256SUMS-$ReleaseTag.txt"

$Products = @(
    @{ Source = 'apps\ninfer.exe'; Destination = 'ninfer.exe' },
    @{ Source = 'apps\ninfer-serve.exe'; Destination = 'ninfer-serve.exe' },
    @{ Source = 'apps\ninfer-perplexity.exe'; Destination = 'ninfer-perplexity.exe' },
    @{ Source = 'bench\ninfer_bench.exe'; Destination = 'ninfer_bench.exe' }
)

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
$resolvedDist = (Resolve-Path -LiteralPath $DistRoot).Path
$resolvedProductParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $ProductRoot))
if ($resolvedProductParent -ne $resolvedDist -or (Split-Path -Leaf $ProductRoot) -ne $ProductName) {
    throw "Refusing to package outside the expected dist directory: $ProductRoot"
}
if (Test-Path -LiteralPath $ProductRoot) {
    Remove-Item -LiteralPath $ProductRoot -Recurse -Force
}
if (Test-Path -LiteralPath $ArchivePath) {
    Remove-Item -LiteralPath $ArchivePath -Force
}
New-Item -ItemType Directory -Path $ProductRoot | Out-Null

foreach ($product in $Products) {
    $source = Join-Path $BuildRoot $product.Source
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing release product: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $ProductRoot $product.Destination)
}
Get-ChildItem -LiteralPath (Join-Path $BuildRoot 'apps') -Filter '*.dll' | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $ProductRoot
}
Copy-Item -LiteralPath (Join-Path $RepoRoot 'VERSION') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'docs\rtx-3090-windows.md') -Destination (Join-Path $ProductRoot 'README.md')
Copy-Item -LiteralPath (Join-Path $RepoRoot 'scripts\download-qwen38.bat') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'scripts\run-qwen38-c1.bat') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'scripts\run-qwen38-c8.bat') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'scripts\run-qwen38-vision.bat') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'scripts\download-qwen36-35b-vision.bat') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'scripts\run-qwen36-35b-vision.bat') -Destination $ProductRoot

$innerHashes = Get-ChildItem -LiteralPath $ProductRoot -File | Sort-Object Name | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($_.Name)"
}
$innerHashes | Set-Content -LiteralPath (Join-Path $ProductRoot 'SHA256SUMS.txt') -Encoding ascii
Compress-Archive -LiteralPath $ProductRoot -DestinationPath $ArchivePath -CompressionLevel Optimal
$archiveHash = Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256
"$($archiveHash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $ArchivePath)" |
    Set-Content -LiteralPath $ChecksumPath -Encoding ascii

Get-Item -LiteralPath $ArchivePath, $ChecksumPath |
    Select-Object Name, @{ Name = 'SizeMB'; Expression = { [math]::Round($_.Length / 1MB, 2) } }
