<#
Run the Playwright system tests and collect execution evidence (STP FSPP-MTP02.0, 8.10).

  .\run_tests.ps1                 # all automated cases
  .\run_tests.ps1 -SkipSlow       # skip ST-31, ST-34, ST-50 (large data / timing)
  .\run_tests.ps1 -Select st07    # one case (pytest -k expression)
  .\run_tests.ps1 -Headed         # watch the browser

Creates ..\04_Test_Execution_Results\<date-time>-<commit>\ with report.html,
junit.xml, results.csv, screenshots\, traces of failed tests (playwright\), the application
log and run-info.txt. Test scripts are in tests\; input files in ..\03_Test_Data\.
The application under test is started by the tests on http://127.0.0.1:5055 only.
#>
param([string]$Select = "", [switch]$SkipSlow, [switch]$Headed)

$ErrorActionPreference = "Stop"
$automation = $PSScriptRoot                     # testing-docs\02_Test_Automation
$testingDocs = Split-Path $automation -Parent   # testing-docs
$repo = (Resolve-Path (Join-Path $testingDocs "..")).Path
$python = Join-Path $automation ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Missing $python. Create the venv and install requirements-e2e.txt first." }

$sha = (git -C $repo rev-parse --short HEAD).Trim()
$dirty = if (git -C $repo status --porcelain --untracked-files=no) { "-modified" } else { "" }
$run = Join-Path $testingDocs ("04_Test_Execution_Results\" + (Get-Date -Format "yyyyMMdd-HHmmss") + "-$sha$dirty")
New-Item -ItemType Directory -Force $run | Out-Null
$env:E2E_RUN_DIR = $run

$data = Join-Path $testingDocs "03_Test_Data"
if (-not (Test-Path (Join-Path $data "TD-12_large_40000.csv"))) {
    & (Join-Path $repo ".venv\Scripts\python.exe") -I (Join-Path $data "make_test_data.py")
}

$chrome = (Get-Item "$env:ProgramFiles\Google\Chrome\Application\chrome.exe" -ErrorAction SilentlyContinue).VersionInfo.ProductVersion
$edge = (Get-Item "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe" -ErrorAction SilentlyContinue).VersionInfo.ProductVersion
@(
    "Run started:      $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "Commit:           $sha$dirty",
    "Machine / OS:     $env:COMPUTERNAME / $((Get-CimInstance Win32_OperatingSystem).Caption)",
    "Application:      http://127.0.0.1:5055 (started by the tests, COMPUTE_DEVICE=cpu)",
    "Default browser:  Google Chrome $chrome (channel chrome)",
    "Also used:        Microsoft Edge $edge (ST-39, ST-49); Playwright Chromium (ST-49)",
    "Python (tests):   $(& $python --version)",
    "Packages:         $((& $python -m pip freeze --disable-pip-version-check | Select-String -Pattern '^(playwright|pytest|pytest-playwright|pytest-html)==') -join ', ')",
    "Selection:        $(if ($Select) { $Select } else { 'all' })$(if ($SkipSlow) { ' (slow skipped)' })"
) | Set-Content -Encoding utf8 (Join-Path $run "run-info.txt")

$pytestArgs = @("-m", "pytest",
    "--html=$run\report.html",
    "--junitxml=$run\junit.xml",
    "--output=$run\playwright",
    "--screenshot=only-on-failure",
    "--tracing=retain-on-failure")
if ($SkipSlow) { $pytestArgs += @("-m", "not slow") }
if ($Select) { $pytestArgs += @("-k", $Select) }
if ($Headed) { $pytestArgs += "--headed" }

Push-Location $automation
try { & $python @pytestArgs; $code = $LASTEXITCODE } finally { Pop-Location }
"Run finished:     $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') (pytest exit code $code)" | Add-Content -Encoding utf8 (Join-Path $run "run-info.txt")
Write-Host "Evidence folder: $run"
exit $code
