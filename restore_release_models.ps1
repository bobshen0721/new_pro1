[CmdletBinding()]
param(
    [string]$Repository = "bobshen0721/new_pro1",
    [string]$ReleaseTag = "models-v2",
    [string]$ModelRoot = (Join-Path $PSScriptRoot "models"),
    [string]$ManifestPath = (Join-Path $PSScriptRoot "model-release-manifest.json"),
    [switch]$KeepDownloads
)

$ErrorActionPreference = "Stop"
$modelRootPath = [IO.Path]::GetFullPath($ModelRoot).TrimEnd("\")
New-Item -ItemType Directory -Force -Path $modelRootPath | Out-Null

$cacheRoot = Join-Path $modelRootPath ".release-cache\$ReleaseTag"
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
$baseUrl = "https://github.com/$Repository/releases/download/$ReleaseTag"

if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    $ManifestPath = Join-Path $cacheRoot "model-release-manifest.json"
    Write-Host "Downloading manifest: $baseUrl/model-release-manifest.json"
    Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/model-release-manifest.json" -OutFile $ManifestPath
}

$manifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $ManifestPath | ConvertFrom-Json
if ($manifest.release_tag -ne $ReleaseTag) {
    throw "Manifest tag mismatch. Expected $ReleaseTag, got $($manifest.release_tag)"
}

foreach ($model in $manifest.models) {
    $targetPath = Join-Path $modelRootPath $model.folder
    $isComplete = $true
    foreach ($requiredFile in $model.required_files) {
        if (-not (Test-Path -LiteralPath (Join-Path $targetPath $requiredFile) -PathType Leaf)) {
            $isComplete = $false
            break
        }
    }
    if ($isComplete) {
        Write-Host "Already present; skipping: $($model.folder)"
        continue
    }
    if (Test-Path -LiteralPath $targetPath) {
        throw "Model directory exists but is incomplete. Move it aside and retry: $targetPath"
    }

    $modelCache = Join-Path $cacheRoot $model.folder
    New-Item -ItemType Directory -Force -Path $modelCache | Out-Null
    foreach ($part in $model.parts) {
        $partPath = Join-Path $modelCache $part.asset
        $needsDownload = $true
        if (Test-Path -LiteralPath $partPath -PathType Leaf) {
            $existingHash = (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $needsDownload = $existingHash -ne $part.sha256
        }
        if ($needsDownload) {
            Write-Host "Downloading: $($part.asset)"
            Invoke-WebRequest -UseBasicParsing -Uri "$baseUrl/$($part.asset)" -OutFile $partPath
        }
        $actualHash = (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $part.sha256) {
            throw "Part SHA-256 mismatch: $($part.asset)"
        }
    }

    $archivePath = Join-Path $modelCache $model.archive
    $archive = [IO.File]::Open($archivePath, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        foreach ($part in $model.parts) {
            $partPath = Join-Path $modelCache $part.asset
            $input = [IO.File]::Open($partPath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
            try {
                $input.CopyTo($archive)
            }
            finally {
                $input.Dispose()
            }
        }
    }
    finally {
        $archive.Dispose()
    }

    $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($archiveHash -ne $model.archive_sha256) {
        throw "Reassembled archive SHA-256 mismatch: $($model.archive)"
    }

    Write-Host "Extracting: $($model.folder)"
    & tar.exe -xf $archivePath -C $modelRootPath
    if ($LASTEXITCODE -ne 0) {
        throw "tar extraction failed: $($model.archive)"
    }
    foreach ($requiredFile in $model.required_files) {
        if (-not (Test-Path -LiteralPath (Join-Path $targetPath $requiredFile) -PathType Leaf)) {
            throw "Required file missing after extraction: $($model.folder)\$requiredFile"
        }
    }

    if (-not $KeepDownloads) {
        $resolvedCache = [IO.Path]::GetFullPath($modelCache)
        if (-not $resolvedCache.StartsWith(([IO.Path]::GetFullPath($cacheRoot) + "\"), [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean a path outside the cache root: $resolvedCache"
        }
        Remove-Item -LiteralPath $resolvedCache -Recurse -Force
    }
}

Write-Host "Three public models are ready: $modelRootPath" -ForegroundColor Green
if (-not (Test-Path -LiteralPath (Join-Path $modelRootPath "pyannote-community-1") -PathType Container)) {
    Write-Warning "The release does not mirror gated pyannote Community-1. Accept its Hugging Face conditions, then use download_models.ps1 with your own HF_TOKEN."
}
