$ErrorActionPreference = "Stop"

$ENDPOINT_NAME = "diabetes-endpoint-$env:ENVIRONMENT"

Write-Host "Testing endpoint: $ENDPOINT_NAME"

# Get endpoint URI and key
$ENDPOINT_URI = az ml online-endpoint show `
  --resource-group $env:RESOURCE_GROUP `
  --workspace-name $env:WORKSPACE_NAME `
  --name $ENDPOINT_NAME `
  --query scoring_uri -o tsv

$ENDPOINT_KEY = az ml online-endpoint get-credentials `
  --resource-group $env:RESOURCE_GROUP `
  --workspace-name $env:WORKSPACE_NAME `
  --name $ENDPOINT_NAME `
  --query primaryKey -o tsv

Write-Host "Endpoint URI: $ENDPOINT_URI"

# Test with sample data
if (Test-Path "deploy/sample-request.json") {
    Write-Host "Sending test request..."
    
    $headers = @{
        "Authorization" = "Bearer $ENDPOINT_KEY"
        "Content-Type" = "application/json"
    }
    
    $body = Get-Content -Path "deploy/sample-request.json" -Raw
    
    try {
        $response = Invoke-RestMethod -Uri $ENDPOINT_URI -Method Post -Headers $headers -Body $body
        Write-Host "Response: $($response | ConvertTo-Json -Depth 10)"
        Write-Host "Endpoint test successful!"
    } catch {
        Write-Host "Error testing endpoint: $_"
        exit 1
    }
} else {
    Write-Host "No sample-request.json found. Skipping endpoint test."
}