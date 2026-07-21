param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [string]$Python = (Get-Command python).Source
)

$ErrorActionPreference = "Stop"
$repo = [System.IO.Path]::GetFullPath($RepoRoot)
$python = $Python
$script = Join-Path $repo "pipeline_artifacts\model_comparison_amendment_v2_20260717\scripts\run_model_comparison_training.py"
$protocol = Join-Path $repo "pipeline_artifacts\model_comparison_amendment_v2_20260717\protocol_v5_memorysafe"
$output = Join-Path $repo "result\model_comparison_v2_20260717\development_v5_memorysafe"
$logRoot = Join-Path $repo "result\model_comparison_v2_20260717"
$stdout = Join-Path $logRoot "training_v5_memorysafe_orchestrator.stdout.log"
$stderr = Join-Path $logRoot "training_v5_memorysafe_orchestrator.stderr.log"

Set-Location $repo
try {
    & $python $script `
        --repo-root $repo `
        --protocol-root $protocol `
        --output-root (Join-Path $output "c3d_all_seeds") `
        --mode train `
        --patience 5 `
        --model-id c3d `
        --seed 50900 `
        --seed 50901 `
        --seed 50902 1>> $stdout 2>> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "C3D training orchestrator exited with code $LASTEXITCODE"
    }
    & $python $script `
        --repo-root $repo `
        --protocol-root $protocol `
        --output-root (Join-Path $output "remaining_seeds23") `
        --mode train `
        --patience 5 `
        --model-id i3d `
        --model-id resnet_lstm `
        --model-id slowfast `
        --model-id swin3d `
        --model-id josenet `
        --seed 50901 `
        --seed 50902 1>> $stdout 2>> $stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Remaining seed-2/3 orchestrator exited with code $LASTEXITCODE"
    }
    exit 0
}
catch {
    $_ | Out-String | Add-Content -Path $stderr -Encoding UTF8
    exit 1
}
