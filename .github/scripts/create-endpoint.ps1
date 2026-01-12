$ErrorActionPreference = "Stop"

$ENDPOINT_NAME = "diabetes-endpoint-$env:ENVIRONMENT"

Write-Host "Creating endpoint: $ENDPOINT_NAME"

# Check if endpoint exists
$endpointExists = az ml online-endpoint list `
  --resource-group $env:RESOURCE_GROUP `
  --workspace-name $env:WORKSPACE_NAME `
  --query "[?name=='$ENDPOINT_NAME'].name" -o tsv

if ([string]::IsNullOrEmpty($endpointExists)) {
    Write-Host "Endpoint does not exist. Creating..."
    
    # Create endpoint using YAML config
    az ml online-endpoint create `
      --resource-group $env:RESOURCE_GROUP `
      --workspace-name $env:WORKSPACE_NAME `
      --file deploy/endpoint.yml `
      --name $ENDPOINT_NAME
    
    Write-Host "Endpoint created successfully"
} else {
    Write-Host "Endpoint already exists: $ENDPOINT_NAME"
}

# Save endpoint name for subsequent steps
"ENDPOINT_NAME=$ENDPOINT_NAME" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append