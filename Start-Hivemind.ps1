[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$SkipBuild,
    [switch]$SkipVoiceBuild,
    [switch]$SkipFolderGuardianBridgeBuild,
    [switch]$PrepareOnly,
    [switch]$PreloadPhi3
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$UiDir = Join-Path $Root "AICoreClient\ui"
$ElectronDir = Join-Path $Root "AICoreClient\electron"
$VoiceProject = Join-Path $Root "VoiceBridge\VoiceBridge.csproj"
$FolderGuardianBridgeProject = Join-Path $Root "FolderGuardianBridge\FolderGuardianBridge.csproj"
$EnvFile = Join-Path $Root ".env"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Get-FirstCommand {
    param([string[]]$Names)

    foreach ($Name in $Names) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue
        if ($Command) {
            return $Command.Source
        }
    }

    return $null
}

function Test-PythonLaunch {
    param(
        [string]$Command,
        [string[]]$Arguments = @()
    )

    if (-not $Command) {
        return $false
    }

    try {
        & $Command @Arguments -c "import sys" *> $null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Get-MissingPythonModules {
    param(
        [string]$Command,
        [string[]]$Arguments,
        [string[]]$Modules
    )

    $ModuleCsv = $Modules -join ","
    $CheckCode = "import importlib.util; mods='$ModuleCsv'.split(','); print(','.join([m for m in mods if importlib.util.find_spec(m) is None]))"
    try {
        $Output = & $Command @Arguments -c $CheckCode 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $Modules
        }

        $Text = ($Output | Out-String).Trim()
        if (-not $Text) {
            return @()
        }

        return $Text -split "," | Where-Object { $_ }
    }
    catch {
        return $Modules
    }
}

function Select-PythonLaunch {
    $ConfiguredPython = [Environment]::GetEnvironmentVariable("HIVEMIND_PYTHON", "Process")
    $ConfiguredArgs = [Environment]::GetEnvironmentVariable("HIVEMIND_PYTHON_ARGS", "Process")

    if ($ConfiguredPython) {
        $Args = if ($ConfiguredArgs) { $ConfiguredArgs -split "\s+" } else { @() }
        if (Test-PythonLaunch -Command $ConfiguredPython -Arguments $Args) {
            return @{ Command = $ConfiguredPython; Arguments = $Args }
        }

        Write-Warning "Configured HIVEMIND_PYTHON did not run successfully: $ConfiguredPython $ConfiguredArgs"
        if (Test-Path -LiteralPath $ConfiguredPython) {
            Write-Warning "Configured Python exists, so the launcher will still use it. This can happen when a sandbox cannot execute user-profile programs."
            return @{ Command = $ConfiguredPython; Arguments = $Args }
        }
    }

    $Candidates = @()

    $Python = Get-FirstCommand @("python.exe", "python")
    if ($Python) {
        $Candidates += @{ Command = $Python; Arguments = @() }
    }

    $PyLauncher = Get-FirstCommand @("py.exe", "py")
    if ($PyLauncher) {
        $Candidates += @{ Command = $PyLauncher; Arguments = @("-3") }
    }

    $RepoVenvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $RepoVenvPython) {
        $Candidates += @{ Command = $RepoVenvPython; Arguments = @() }
    }

    foreach ($Candidate in $Candidates) {
        if (Test-PythonLaunch -Command $Candidate.Command -Arguments $Candidate.Arguments) {
            return $Candidate
        }
    }

    return $null
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Import-EnvFile {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    foreach ($Line in Get-Content -LiteralPath $Path) {
        if ($Line -match '^\s*#' -or $Line -notmatch '^\s*([^=\s]+)\s*=\s*(.*)\s*$') {
            continue
        }

        $Name = $Matches[1]
        $Value = $Matches[2].Trim()
        $Value = $Value -replace '^["'']|["'']$', ''

        if (-not [Environment]::GetEnvironmentVariable($Name, "Process")) {
            [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
        }
    }
}

function Ensure-NodeModules {
    param(
        [string]$ProjectDir,
        [string]$ProjectName,
        [string]$Npm
    )

    $NodeModules = Join-Path $ProjectDir "node_modules"
    if (Test-Path -LiteralPath $NodeModules) {
        return
    }

    if (-not $Install) {
        throw "$ProjectName dependencies are missing. Run .\Start-Hivemind.ps1 -Install once, or double-click Start-Hivemind.bat."
    }

    Write-Step "Installing $ProjectName dependencies"
    Invoke-Checked -FilePath $Npm -Arguments @("install") -WorkingDirectory $ProjectDir
}

function Get-NewestExistingFile {
    param([string[]]$Paths)

    $Files = foreach ($Path in $Paths) {
        if (Test-Path -LiteralPath $Path) {
            Get-Item -LiteralPath $Path
        }
    }

    return $Files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

function Get-NewestVoiceSource {
    $VoiceDir = Join-Path $Root "VoiceBridge"
    return Get-ChildItem -LiteralPath $VoiceDir -Recurse -File -Include *.cs,*.csproj |
        Where-Object { $_.FullName -notmatch '\\(bin|obj)\\' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Get-FolderGuardianRoot {
    $FolderGuardianRoot = [Environment]::GetEnvironmentVariable("HIVEMIND_FOLDERGUARDIAN_ROOT", "Process")
    if (-not $FolderGuardianRoot) {
        $FolderGuardianRoot = Join-Path $Root "external\FolderGuardian"
    }

    return $FolderGuardianRoot
}

function Get-NewestFolderGuardianBridgeSource {
    $BridgeDir = Join-Path $Root "FolderGuardianBridge"
    $Files = @()
    if (Test-Path -LiteralPath $BridgeDir) {
        $Files += Get-ChildItem -LiteralPath $BridgeDir -Recurse -File -Include *.cs,*.csproj |
            Where-Object { $_.FullName -notmatch '\\(bin|obj)\\' }
    }

    $FolderGuardianRoot = Get-FolderGuardianRoot
    $FolderGuardianCore = Join-Path $FolderGuardianRoot "Core"
    if (Test-Path -LiteralPath $FolderGuardianCore) {
        $Files += Get-ChildItem -LiteralPath $FolderGuardianCore -File -Filter *.cs
    }

    return $Files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

Set-Location $Root
Import-EnvFile -Path $EnvFile

if ($PreloadPhi3) {
    $env:HIVEMIND_EAGER_LOAD_PHI3 = "true"
}

$Npm = Get-FirstCommand @("npm.cmd", "npm")
if (-not $Npm) {
    throw "npm was not found. Install Node.js, then run Start-Hivemind again."
}

$SelectedPython = Select-PythonLaunch
if ($SelectedPython) {
    $env:HIVEMIND_PYTHON = $SelectedPython.Command
    $env:HIVEMIND_PYTHON_ARGS = $SelectedPython.Arguments -join " "

    if ($SelectedPython.Command -eq (Join-Path $Root ".venv\Scripts\python.exe")) {
        Write-Warning "Using the repo .venv Python. If the backend reports missing AI/voice packages, point HIVEMIND_PYTHON in .env at your full AI Python environment."
    }

    $FullBackendModules = @(
        "grpc",
        "transformers",
        "mistralai",
        "torch",
        "accelerate",
        "pyttsx3",
        "win32com",
        "requests",
        "geocoder",
        "gtts",
        "pydub",
        "langdetect",
        "speech_recognition"
    )
    $MissingModules = @(Get-MissingPythonModules -Command $SelectedPython.Command -Arguments $SelectedPython.Arguments -Modules $FullBackendModules)
    if ($MissingModules.Count -gt 0 -and -not $env:HIVEMIND_BACKEND_SCRIPT) {
        $env:HIVEMIND_BACKEND_SCRIPT = "AICoreClient\PythonServer\runtime_lite_server.py"
        Write-Warning "Full AI backend dependencies are missing ($($MissingModules -join ', ')). Starting runtime-lite backend so the app can still open and test runtime tools."
    }
}
else {
    Write-Warning "No runnable Python was found. Set HIVEMIND_PYTHON in .env to the backend Python executable."
}

Ensure-NodeModules -ProjectDir $UiDir -ProjectName "React UI" -Npm $Npm
Ensure-NodeModules -ProjectDir $ElectronDir -ProjectName "Electron shell" -Npm $Npm

if (-not $SkipBuild) {
    Write-Step "Building React UI"
    Invoke-Checked -FilePath $Npm -Arguments @("run", "build") -WorkingDirectory $UiDir
}

$VoiceCandidates = @(
    (Join-Path $Root "VoiceBridge\bin\Release\net8.0\VoiceBridge.exe"),
    (Join-Path $Root "VoiceBridge\bin\Debug\net8.0\VoiceBridge.exe")
)
$VoiceExe = Get-NewestExistingFile -Paths $VoiceCandidates
$VoiceSource = Get-NewestVoiceSource

if (-not $SkipVoiceBuild -and (Test-Path -LiteralPath $VoiceProject) -and ((-not $VoiceExe) -or ($VoiceSource -and $VoiceSource.LastWriteTime -gt $VoiceExe.LastWriteTime))) {
    $Dotnet = Get-FirstCommand @("dotnet.exe", "dotnet")
    if ($Dotnet) {
        Write-Step "Building VoiceBridge"
        try {
            Invoke-Checked -FilePath $Dotnet -Arguments @("build", $VoiceProject) -WorkingDirectory $Root
            $VoiceExe = Get-NewestExistingFile -Paths $VoiceCandidates
        }
        catch {
            if ($VoiceExe) {
                Write-Warning "VoiceBridge rebuild failed, so the launcher will use the existing executable: $($VoiceExe.FullName)"
                Write-Warning $_.Exception.Message
            }
            else {
                throw
            }
        }
    }
    else {
        Write-Warning ".NET SDK was not found. Voice input will be unavailable until VoiceBridge is built."
    }
}

if ($VoiceExe) {
    $env:HIVEMIND_VOICEBRIDGE_EXE = $VoiceExe.FullName
    Write-Host "VoiceBridge: $($VoiceExe.FullName)"
}

$FolderGuardianBridgeCandidates = @(
    (Join-Path $Root "FolderGuardianBridge\bin\Release\net8.0-windows\FolderGuardianBridge.exe"),
    (Join-Path $Root "FolderGuardianBridge\bin\Debug\net8.0-windows\FolderGuardianBridge.exe")
)
$FolderGuardianBridgeExe = Get-NewestExistingFile -Paths $FolderGuardianBridgeCandidates
$FolderGuardianBridgeSource = Get-NewestFolderGuardianBridgeSource
$FolderGuardianRoot = Get-FolderGuardianRoot
$FolderGuardianCore = Join-Path $FolderGuardianRoot "Core"
$CanBuildFolderGuardianBridge = Test-Path -LiteralPath $FolderGuardianCore

if (-not $CanBuildFolderGuardianBridge -and -not $FolderGuardianBridgeExe) {
    Write-Warning "FolderGuardian core was not found at '$FolderGuardianCore'. Set HIVEMIND_FOLDERGUARDIAN_ROOT in .env to enable live encrypt/decrypt."
}
elseif (-not $SkipFolderGuardianBridgeBuild -and (Test-Path -LiteralPath $FolderGuardianBridgeProject) -and ((-not $FolderGuardianBridgeExe) -or ($FolderGuardianBridgeSource -and $FolderGuardianBridgeSource.LastWriteTime -gt $FolderGuardianBridgeExe.LastWriteTime))) {
    $Dotnet = Get-FirstCommand @("dotnet.exe", "dotnet")
    if ($Dotnet) {
        Write-Step "Building FolderGuardianBridge"
        try {
            Invoke-Checked -FilePath $Dotnet -Arguments @("build", $FolderGuardianBridgeProject) -WorkingDirectory $Root
            $FolderGuardianBridgeExe = Get-NewestExistingFile -Paths $FolderGuardianBridgeCandidates
        }
        catch {
            if ($FolderGuardianBridgeExe) {
                Write-Warning "FolderGuardianBridge rebuild failed, so the launcher will use the existing executable: $($FolderGuardianBridgeExe.FullName)"
                Write-Warning $_.Exception.Message
            }
            else {
                throw
            }
        }
    }
    else {
        Write-Warning ".NET SDK was not found. FolderGuardian live encrypt/decrypt will be unavailable until FolderGuardianBridge is built."
    }
}

if ($FolderGuardianBridgeExe) {
    $env:HIVEMIND_FOLDERGUARDIAN_BRIDGE_EXE = $FolderGuardianBridgeExe.FullName
    Write-Host "FolderGuardianBridge: $($FolderGuardianBridgeExe.FullName)"
}

if ($env:HIVEMIND_PYTHON) {
    $PythonLine = $env:HIVEMIND_PYTHON
    if ($env:HIVEMIND_PYTHON_ARGS) {
        $PythonLine = "$PythonLine $env:HIVEMIND_PYTHON_ARGS"
    }
    Write-Host "Python: $PythonLine"
}

if ($env:HIVEMIND_BACKEND_SCRIPT) {
    Write-Host "Backend: $env:HIVEMIND_BACKEND_SCRIPT"
}

if ($PrepareOnly) {
    Write-Step "Hivemind launch preparation complete"
    return
}

Write-Step "Starting Hivemind"
Invoke-Checked -FilePath $Npm -Arguments @("start") -WorkingDirectory $ElectronDir
