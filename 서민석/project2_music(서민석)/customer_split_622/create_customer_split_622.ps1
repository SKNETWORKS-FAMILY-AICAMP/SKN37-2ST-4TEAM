# 원본 CSV를 읽기 전용으로 사용해 80/10/10 층화 무작위 분할 파일을 만듭니다.
$ErrorActionPreference = "Stop"

$OutputDirectory = $PSScriptRoot
$DataDirectory = Join-Path $OutputDirectory "data"
$null = New-Item -ItemType Directory -Path $DataDirectory -Force
$ProjectRoot = Split-Path -Parent $OutputDirectory
$InputPath = Join-Path $ProjectRoot "music_churn_data_v1\data\processed\customer_features_20181101_v1.csv"
$Seed = 2027

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "원본 CSV를 찾을 수 없습니다: $InputPath"
}

$Rows = @(Import-Csv -LiteralPath $InputPath)
if ($Rows.Count -eq 0) {
    throw "원본 CSV에 데이터가 없습니다."
}
$Columns = @($Rows[0].PSObject.Properties.Name)
if ($Columns -notcontains "user_id" -or $Columns -notcontains "churn") {
    throw "원본 CSV에 user_id와 churn 컬럼이 필요합니다."
}
if ($Columns -contains "split") {
    throw "입력 CSV에 split 컬럼이 이미 있습니다. 분할 없는 원본 파일을 사용해 주세요."
}
if (@($Rows | Select-Object -ExpandProperty user_id -Unique).Count -ne $Rows.Count) {
    throw "원본 CSV에 중복 user_id가 있습니다."
}

function Shuffle-Rows {
    param([object[]]$Items, [System.Random]$Generator)
    $Shuffled = [object[]]$Items.Clone()
    for ($Index = $Shuffled.Length - 1; $Index -gt 0; $Index--) {
        $SwapIndex = $Generator.Next(0, $Index + 1)
        $Temporary = $Shuffled[$Index]
        $Shuffled[$Index] = $Shuffled[$SwapIndex]
        $Shuffled[$SwapIndex] = $Temporary
    }
    return $Shuffled
}

# 각 churn 클래스 안에서 따로 섞어 이탈 비율이 train/valid/test에 유지되게 합니다.
$Generator = [System.Random]::new($Seed)
$Assignments = [System.Collections.Generic.List[object]]::new()
foreach ($ClassName in @("0", "1")) {
    $ClassRows = @($Rows | Where-Object { [string]$_.churn -eq $ClassName })
    if ($ClassRows.Count -eq 0) {
        throw "churn=$ClassName 행이 없습니다."
    }
    $ShuffledRows = @(Shuffle-Rows -Items $ClassRows -Generator $Generator)
    $TrainCount = [int][Math]::Floor($ShuffledRows.Count * 0.60)
    $RemainingCount = $ShuffledRows.Count - $TrainCount
    $ValidCount = [int][Math]::Floor($ShuffledRows.Count * 0.20)

    for ($Index = 0; $Index -lt $ShuffledRows.Count; $Index++) {
        if ($Index -lt $TrainCount) {
            $SplitName = "train"
        } elseif ($Index -lt ($TrainCount + $ValidCount)) {
            $SplitName = "valid"
        } else {
            $SplitName = "test"
        }
        $Assignments.Add([pscustomobject]@{ Row = $ShuffledRows[$Index]; Split = $SplitName })
    }
}

$Manifest = foreach ($Assignment in $Assignments) {
    [pscustomobject]@{
        user_id = $Assignment.Row.user_id
        churn = $Assignment.Row.churn
        split = $Assignment.Split
    }
}
$Manifest | Export-Csv -LiteralPath (Join-Path $DataDirectory "customer_split_manifest_622.csv") -NoTypeInformation -Encoding utf8

foreach ($SplitName in @("train", "valid", "test")) {
    $SplitRows = @($Assignments | Where-Object { $_.Split -eq $SplitName } | ForEach-Object { $_.Row })
    $Destination = Join-Path $DataDirectory "${SplitName}_20181101_622.csv"
    $SplitRows | Select-Object -Property $Columns | Export-Csv -LiteralPath $Destination -NoTypeInformation -Encoding utf8
}

$CombinedRows = foreach ($Assignment in $Assignments) {
    $OutputRow = [ordered]@{}
    foreach ($Column in $Columns) {
        $OutputRow[$Column] = $Assignment.Row.$Column
    }
    $OutputRow["split"] = $Assignment.Split
    [pscustomobject]$OutputRow
}
$CombinedRows | Export-Csv -LiteralPath (Join-Path $DataDirectory "customer_features_20181101_622_with_split.csv") -NoTypeInformation -Encoding utf8

$Summary = foreach ($SplitName in @("train", "valid", "test")) {
    $Subset = @($Assignments | Where-Object { $_.Split -eq $SplitName })
    $PositiveCount = @($Subset | Where-Object { [string]$_.Row.churn -eq "1" }).Count
    [pscustomobject]@{
        Split = $SplitName
        Customers = $Subset.Count
        ChurnCustomers = $PositiveCount
        ChurnRate = if ($Subset.Count -gt 0) { [Math]::Round($PositiveCount / $Subset.Count, 5) } else { 0 }
    }
}
$Summary | Format-Table -AutoSize
Write-Host "Created split files in: $DataDirectory"
Write-Host "Source was not modified: $InputPath"
