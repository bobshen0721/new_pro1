[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ModelRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot,

    [ValidateRange(1, 1900)]
    [int]$PartSizeMiB = 1900,

    [string[]]$IncludeFolders
)

$ErrorActionPreference = "Stop"

$allModels = @(
    [PSCustomObject]@{
        Folder = "faster-whisper-large-v2"
        Source = "https://huggingface.co/Systran/faster-whisper-large-v2"
        Revision = "f0fe81560cb8b68660e564f55dd99207059c092e"
        License = "MIT"
        RequiredFiles = @("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")
    },
    [PSCustomObject]@{
        Folder = "TEA-ASR-1.1-mini"
        Source = "https://huggingface.co/JacobLinCool/TEA-ASR-1.1-mini"
        Revision = "98c58048572b44839dfcfa60de3ad7e365a5b232"
        License = "MIT; underlying Qwen weights remain Apache-2.0"
        RequiredFiles = @(
            "added_tokens.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json"
        )
    },
    [PSCustomObject]@{
        Folder = "Qwen3-ForcedAligner-0.6B"
        Source = "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B"
        Revision = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
        License = "Apache-2.0"
        RequiredFiles = @(
            "config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "vocab.json"
        )
    }
)

if ($IncludeFolders) {
    $knownFolders = @($allModels | ForEach-Object { $_.Folder })
    $unknownFolders = @($IncludeFolders | Where-Object { $_ -notin $knownFolders })
    if ($unknownFolders.Count -gt 0) {
        throw "Unknown model folder(s): $($unknownFolders -join ', ')"
    }
    $models = @($allModels | Where-Object { $_.Folder -in $IncludeFolders })
}
else {
    $models = $allModels
}

$modelRootPath = [IO.Path]::GetFullPath($ModelRoot).TrimEnd("\")
if (-not (Test-Path -LiteralPath $modelRootPath -PathType Container)) {
    throw "Model root not found: $modelRootPath"
}

$outputRootPath = [IO.Path]::GetFullPath($OutputRoot).TrimEnd("\")
New-Item -ItemType Directory -Force -Path $outputRootPath | Out-Null
if (Get-ChildItem -LiteralPath $outputRootPath -Force -ErrorAction SilentlyContinue) {
    throw "Output directory must be empty: $outputRootPath"
}

$tar = Get-Command tar.exe -ErrorAction Stop
$partSizeBytes = [int64]$PartSizeMiB * 1MB
$buffer = New-Object byte[] (4MB)
$results = @()

foreach ($model in $models) {
    $sourcePath = Join-Path $modelRootPath $model.Folder
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Container)) {
        throw "Model directory not found: $sourcePath"
    }
    foreach ($requiredFile in $model.RequiredFiles) {
        $requiredPath = Join-Path $sourcePath $requiredFile
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required file missing: $requiredPath"
        }
    }

    $archiveName = "$($model.Folder).tar"
    $archivePath = Join-Path $outputRootPath $archiveName
    Write-Host "Creating archive: $archiveName"
    & $tar.Source -cf $archivePath "--exclude=$($model.Folder)/.cache" -C $modelRootPath $model.Folder
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed for model: $($model.Folder)"
    }

    $archiveInfo = Get-Item -LiteralPath $archivePath
    $archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $parts = @()
    $input = [IO.File]::Open($archivePath, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        $partNumber = 1
        while ($input.Position -lt $input.Length) {
            $partName = "$archiveName.part{0:D3}" -f $partNumber
            $partPath = Join-Path $outputRootPath $partName
            $output = [IO.File]::Open($partPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
            try {
                $remaining = [Math]::Min($partSizeBytes, $input.Length - $input.Position)
                while ($remaining -gt 0) {
                    $readSize = [int][Math]::Min($buffer.Length, $remaining)
                    $read = $input.Read($buffer, 0, $readSize)
                    if ($read -le 0) {
                        throw "Unexpected end of archive: $archivePath"
                    }
                    $output.Write($buffer, 0, $read)
                    $remaining -= $read
                }
            }
            finally {
                $output.Dispose()
            }

            $partInfo = Get-Item -LiteralPath $partPath
            $parts += [PSCustomObject]@{
                asset = $partName
                size_bytes = $partInfo.Length
                sha256 = (Get-FileHash -LiteralPath $partPath -Algorithm SHA256).Hash.ToLowerInvariant()
            }
            Write-Host "  Created part: $partName ($([Math]::Round($partInfo.Length / 1MB, 1)) MiB)"
            $partNumber++
        }
    }
    finally {
        $input.Dispose()
    }

    $results += [PSCustomObject]@{
        folder = $model.Folder
        source = $model.Source
        revision = $model.Revision
        license = $model.License
        archive = $archiveName
        archive_size_bytes = $archiveInfo.Length
        archive_sha256 = $archiveHash
        required_files = @($model.RequiredFiles)
        parts = $parts
    }

    Remove-Item -LiteralPath $archivePath -Force
}

$resultPath = Join-Path $outputRootPath "packaging-result.json"
$results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $resultPath -Encoding UTF8
Write-Host "Packaging complete: $outputRootPath"
Write-Host "Packaging result: $resultPath"
