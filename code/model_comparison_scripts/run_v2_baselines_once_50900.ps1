param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [string]$Python = (Get-Command python).Source
)

$ErrorActionPreference = 'Stop'
$artifactRoot = Join-Path $RepoRoot 'pipeline_artifacts\model_comparison_amendment_v2_20260717'
$resultRoot = Join-Path $RepoRoot 'result\model_comparison_v2_20260717\test_v2_once'
$statusPath = Join-Path $artifactRoot 'v2_baseline_runner_status.json'
$stdoutPath = Join-Path $artifactRoot 'v2_baseline_runner.stdout.log'
$stderrPath = Join-Path $artifactRoot 'v2_baseline_runner.stderr.log'
$attemptPath = Join-Path $resultRoot 'V2_TEST_ATTEMPT.json'

function Write-Status([string]$State, [int]$ExitCode, [object]$Extra) {
    $payload = [ordered]@{
        state = $State
        exit_code = $ExitCode
        updated_utc = [DateTime]::UtcNow.ToString('o')
        result_root = $resultRoot
        attempt_marker = $attemptPath
    }
    if ($null -ne $Extra) {
        foreach ($property in $Extra.PSObject.Properties) {
            $payload[$property.Name] = $property.Value
        }
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $statusPath
}

try {
    if (Test-Path $attemptPath) {
        $existingAttempt = Get-Content -Raw $attemptPath | ConvertFrom-Json
        if ($existingAttempt.status -eq 'complete' -and [int]$existingAttempt.completed_runs -eq 18) {
            Write-Status 'complete' 0 ([pscustomobject]@{
                attempt_id = $existingAttempt.attempt_id
                completed_runs = $existingAttempt.completed_runs
                idempotent_duplicate_wrapper_launch_refused = $true
                note = 'The wrapper was invoked after the completed one-time attempt; no evaluator process was started.'
            })
            return
        }
        throw "One-time v2 attempt marker already exists with non-complete status: $attemptPath"
    }
    $duplicates = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and $_.CommandLine -match 'evaluate_frozen_baselines_v2_once.py'
    })
    if ($duplicates.Count -gt 0) {
        throw "Duplicate v2 evaluator process detected: $($duplicates.ProcessId -join ',')"
    }

    $telemetry = @()
    1..4 | ForEach-Object {
        $values = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits).Trim().Split(',')
        $telemetry += [ordered]@{
            utilization_percent = [int]$values[0].Trim()
            memory_used_mib = [int]$values[1].Trim()
            memory_total_mib = [int]$values[2].Trim()
            sampled_utc = [DateTime]::UtcNow.ToString('o')
        }
        Start-Sleep -Seconds 2
    }
    Write-Status 'running' -1 ([pscustomobject]@{ preflight_gpu = $telemetry })

    $arguments = @(
        (Join-Path $artifactRoot 'scripts\evaluate_frozen_baselines_v2_once.py'),
        '--repo-root', $RepoRoot,
        '--protocol-root', (Join-Path $artifactRoot 'protocol_v5_memorysafe'),
        '--validation-freeze-root', (Join-Path $artifactRoot 'validation_freeze_v1'),
        '--output-root', $resultRoot,
        '--device', 'cuda'
    )
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $RepoRoot -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    if ($process.ExitCode -ne 0) {
        throw "Evaluator exited with code $($process.ExitCode); see $stderrPath"
    }
    $attempt = Get-Content -Raw $attemptPath | ConvertFrom-Json
    if ($attempt.status -ne 'complete' -or [int]$attempt.completed_runs -ne 18) {
        throw 'Evaluator returned zero but the one-time marker is not complete for 18 runs.'
    }
    Write-Status 'complete' 0 ([pscustomobject]@{ attempt_id = $attempt.attempt_id; completed_runs = $attempt.completed_runs })
}
catch {
    Write-Status 'failed' 1 ([pscustomobject]@{ error = $_.Exception.Message })
    throw
}
