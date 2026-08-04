[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AssetRoot,

    [Parameter(Mandatory = $true)]
    [string]$ManifestPath
)

$ErrorActionPreference = "Stop"
$assetRootPath = [IO.Path]::GetFullPath($AssetRoot).TrimEnd("\")
if (-not (Test-Path -LiteralPath $assetRootPath -PathType Container)) {
    throw "Asset root not found: $assetRootPath"
}

$manifestFile = [IO.Path]::GetFullPath($ManifestPath)
$manifest = Get-Content -Raw -Encoding UTF8 -LiteralPath $manifestFile | ConvertFrom-Json
$validationRoot = Join-Path $assetRootPath (".validation-" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $validationRoot | Out-Null

try {
    foreach ($model in $manifest.models) {
        $archivePath = Join-Path $validationRoot $model.archive
        $archive = [IO.File]::Open($archivePath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        try {
            foreach ($part in $model.parts) {
                $partPath = Join-Path $assetRootPath $part.asset
                if (-not (Test-Path -LiteralPath $partPath -PathType Leaf)) {
                    throw "Missing part: $partPath"
                }
                $partInfo = Get-Item -LiteralPath $partPath
                if ($partInfo.Length -ne [int64]$part.size_bytes) {
                    throw "Part size mismatch: $($part.asset)"
                }
                $partHash = (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($partHash -ne $part.sha256) {
                    throw "Part SHA-256 mismatch: $($part.asset)"
                }

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

        $archiveInfo = Get-Item -LiteralPath $archivePath
        if ($archiveInfo.Length -ne [int64]$model.archive_size_bytes) {
            throw "Archive size mismatch: $($model.archive)"
        }
        $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($archiveHash -ne $model.archive_sha256) {
            throw "Archive SHA-256 mismatch: $($model.archive)"
        }

        $entries = @(& tar.exe -tf $archivePath)
        if ($LASTEXITCODE -ne 0) {
            throw "Unable to list archive: $($model.archive)"
        }
        foreach ($requiredFile in $model.required_files) {
            $requiredEntry = "$($model.folder)/$($requiredFile.Replace('\', '/'))"
            if ($entries -notcontains $requiredEntry) {
                throw "Required archive entry missing: $requiredEntry"
            }
        }
        Write-Host "Validated: $($model.archive)"
        Remove-Item -LiteralPath $archivePath -Force
    }
}
finally {
    if (Test-Path -LiteralPath $validationRoot) {
        $resolvedValidationRoot = [IO.Path]::GetFullPath($validationRoot)
        if (-not $resolvedValidationRoot.StartsWith(($assetRootPath + "\"), [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to clean a path outside the asset root: $resolvedValidationRoot"
        }
        Remove-Item -LiteralPath $resolvedValidationRoot -Recurse -Force
    }
}

Write-Host "All release assets passed validation." -ForegroundColor Green
