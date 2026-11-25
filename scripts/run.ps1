param([switch]$Stop, [switch]$Rebuild)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$statePath = Join-Path $root 'data\run.json'

if ($Stop) {
    if (-not (Test-Path -LiteralPath $statePath)) { Write-Host 'No CCTV run record found.'; exit 0 }
    $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
    $server = Get-Process -Id $state.server -ErrorAction SilentlyContinue
    if ($server -and $server.ProcessName -eq 'go2rtc' -and $server.Path -eq $state.serverPath) {
        Stop-Process -Id $server.Id -Force
    }
    $worker = Get-Process -Id $state.worker -ErrorAction SilentlyContinue
    if ($worker -and $worker.ProcessName -eq 'python' -and $worker.Path -eq $state.workerPath) {
        $details = Get-CimInstance Win32_Process -Filter "ProcessId = $($state.worker)" -ErrorAction SilentlyContinue
        if ($details.CommandLine -match 'cctv_ai\.worker') { Stop-Process -Id $worker.Id -Force }
    }
    Remove-Item -LiteralPath $statePath -Force
    Write-Host 'Stopped the CCTV server and worker processes started by this script.'
    exit 0
}

if (-not (Test-Path -LiteralPath (Join-Path $root '.env'))) {
    throw 'Create .env from .env.example and set your camera RTSP URLs before starting.'
}
$python = Join-Path $root 'ai\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Create ai\.venv and install ai\requirements.txt first.' }
Push-Location $root
try {
    & $python -c 'import cv2, numpy, onnxruntime, rapidocr_onnxruntime, rapidfuzz, yaml'
    if ($LASTEXITCODE -ne 0) { throw 'AI dependencies are incomplete; install ai\requirements.txt.' }

    $listener = Get-NetTCPConnection -LocalPort 1984 -State Listen -ErrorAction SilentlyContinue
    if ($listener) { throw 'Port 1984 is already in use. Run scripts\run.ps1 -Stop first if it is the previous CCTV instance.' }
    if ($Rebuild -or -not (Test-Path -LiteralPath (Join-Path $root 'go2rtc.exe'))) {
        & go build -o go2rtc.exe .
        if ($LASTEXITCODE -ne 0) { throw 'Go server build failed.' }
    }

    $server = Start-Process -FilePath (Join-Path $root 'go2rtc.exe') -WorkingDirectory $root -WindowStyle Hidden -PassThru
    $worker = Start-Process -FilePath $python -ArgumentList @('-m', 'cctv_ai.worker') -WorkingDirectory (Join-Path $root 'ai') -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddSeconds(35)
    $healthy = $false
    while ((Get-Date) -lt $deadline) {
        $probe = & curl.exe --noproxy '*' --max-time 2 --silent --output NUL --write-out '%{http_code}' 'http://127.0.0.1:1984/api/cctv/status' 2>$null
        if ($probe -eq '200') { $healthy = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $healthy) {
        Stop-Process -Id $worker.Id,$server.Id -Force -ErrorAction SilentlyContinue
        throw 'The CCTV server did not become ready on port 1984.'
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $statePath) | Out-Null
    @{server=$server.Id; worker=$worker.Id; serverPath=$server.Path; workerPath=$worker.Path} | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
    Write-Host 'CCTV website: http://localhost:1984/cctv.html'
    Write-Host 'Health:       http://localhost:1984/api/cctv/health'
    Write-Host 'Stop:         .\scripts\run.ps1 -Stop'
} finally {
    Pop-Location
}
