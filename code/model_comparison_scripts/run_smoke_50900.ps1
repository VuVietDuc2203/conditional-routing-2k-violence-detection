param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [string]$Python = (Get-Command python).Source
)

$ErrorActionPreference = "Stop"
$repo = [System.IO.Path]::GetFullPath($RepoRoot)
$python = $Python
$script = Join-Path $repo "pipeline_artifacts\model_comparison_amendment_v2_20260717\scripts\run_model_comparison_training.py"
$protocol = Join-Path $repo "pipeline_artifacts\model_comparison_amendment_v2_20260717\protocol_v2"
$output = Join-Path $repo "result\model_comparison_v2_20260717\smoke_final"
$logRoot = Join-Path $repo "result\model_comparison_v2_20260717"
$stdout = Join-Path $logRoot "smoke_final_orchestrator.stdout.log"
$stderr = Join-Path $logRoot "smoke_final_orchestrator.stderr.log"

Set-Location $repo
try {
    & $python $script `
        --repo-root $repo `
        --protocol-root $protocol `
        --output-root $output `
        --mode smoke 1>> $stdout 2>> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke orchestrator exited with code $LASTEXITCODE"
    }
    exit 0
}
catch {
    $_ | Out-String | Add-Content -Path $stderr -Encoding UTF8
    exit 1
}
