#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Server,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$Draft,
    [Parameter(Mandatory=$true)][string]$Projector,
    [ValidateSet('Baseline', 'NoCheckpoints', 'Static')][string]$Mode = 'Baseline',
    [int]$Port = 8099,
    [string]$OutputRoot = '.',
    [string[]]$ExtraArguments = @()
)

$ErrorActionPreference = 'Stop'
$serverPath = (Resolve-Path -LiteralPath $Server).Path
$modelPath = (Resolve-Path -LiteralPath $Model).Path
$draftPath = (Resolve-Path -LiteralPath $Draft).Path
$projectorPath = (Resolve-Path -LiteralPath $Projector).Path
$runName = 'issue134-' + $Mode + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$runDir = (New-Item -ItemType Directory -Path (Join-Path $OutputRoot $runName)).FullName
$serverArgs = @(
    '--model', $modelPath, '--mmproj', $projectorPath, '--mmproj-device', 'CUDA0',
    '-dev', 'CUDA0,CUDA1', '-sm', 'layer', '-ts', '3.9,3.1', '-ngl', '999',
    '-c', '200000', '-np', '1', '-fa', 'on', '-b', '2048', '-ub', '512',
    '-md', $draftPath, '--spec-type', 'draft-dflash', '--spec-draft-n-max', '7',
    '--spec-draft-p-min', '0.0', '--spec-draft-ngl', '999', '--spec-draft-device', 'CUDA1',
    '--spec-draft-type-k', 'q8_0', '--spec-draft-type-v', 'q8_0',
    '--cache-ram', '0', '--fit', 'off', '--host', '127.0.0.1', '--port', "$Port", '-lv', '5'
)
if ($Mode -eq 'Static') {
    $serverArgs += @('-ctk', 't8', '-ctv', 't4')
} else {
    $serverArgs += @('-ctk', 'vbr', '-ctv', 'vbr', '--vbr-entry', 't8', '--vbr-floor', 't4')
}
$checkpoints = if ($Mode -eq 'NoCheckpoints') { '0' } else { '2' }
$serverArgs += @('--ctx-checkpoints', $checkpoints)
$serverArgs += $ExtraArguments

$oldDiag = [Environment]::GetEnvironmentVariable('GGML_VBR_DIAG', 'Process')
$monitor = $null
$monitorStarted = $false
$exitCode = $null
try {
    $env:GGML_VBR_DIAG = '1'
    $selectedEnv = @{}
    foreach ($name in @('GGML_VBR_DIAG', 'CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING',
        'VBR_GROWTH_HEADROOM_MIB', 'VBR_VRAM_HEADROOM_MIB', 'VBR_FREEZE', 'VBR_BUDGET_MIB', 'VBR_PROMOTE')) {
        $selectedEnv[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    }
    $metadata = [ordered]@{
        utc = [DateTime]::UtcNow.ToString('o'); mode = $Mode
        executable = $serverPath; sha256 = (Get-FileHash -LiteralPath $serverPath -Algorithm SHA256).Hash
        args = $serverArgs; environment = $selectedEnv
        os = [Environment]::OSVersion.VersionString; powershell = $PSVersionTable.PSVersion.ToString()
    }
    $metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runDir 'command.json') -Encoding UTF8
    # Native stderr is informational for version/driver probes, including in Windows PowerShell 5.1.
    $ErrorActionPreference = 'Continue'
    & $serverPath --version 2>&1 | ForEach-Object { "$_" } | Set-Content -LiteralPath (Join-Path $runDir 'version.txt') -Encoding UTF8
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        & $smi.Source -q 2>&1 | ForEach-Object { "$_" } | Set-Content -LiteralPath (Join-Path $runDir 'nvidia-smi.txt') -Encoding UTF8
        $monitor = New-Object System.Diagnostics.Process
        $monitor.StartInfo.FileName = $smi.Source
        $monitor.StartInfo.UseShellExecute = $false
        $monitor.StartInfo.CreateNoWindow = $true
        $gpuLog = Join-Path $runDir 'gpu.csv'
        $monitor.StartInfo.Arguments = '--query-gpu=timestamp,index,uuid,memory.total,memory.used,memory.free,utilization.gpu --format=csv -l 1 --filename="' + $gpuLog + '"'
        $monitorStarted = $monitor.Start()
    } else {
        Write-Warning 'nvidia-smi was not found; GPU telemetry will be missing.'
    }
    Write-Host "Logs: $runDir"
    Write-Host "Send the SAME full failing request to http://127.0.0.1:$Port (bypass the restarting proxy)."
    Write-Host 'This script does not send a prompt, restart the server, or modify your model files.'
    & $serverPath @serverArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $runDir 'server.log')
    $exitCode = $LASTEXITCODE
} finally {
    if ($monitor) {
        if ($monitorStarted -and -not $monitor.HasExited) { $monitor.Kill(); $monitor.WaitForExit() }
        $monitor.Dispose()
    }
    [Environment]::SetEnvironmentVariable('GGML_VBR_DIAG', $oldDiag, 'Process')
    [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); exit_code = $exitCode } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runDir 'result.json') -Encoding UTF8
    Write-Host "Evidence retained in $runDir. Review server.log for private content before sharing."
}
if ($null -ne $exitCode -and $exitCode -ne 0) { exit $exitCode }
