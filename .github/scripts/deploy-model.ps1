$ErrorActionPreference = "Stop"

$ENDPOINT_NAME = "diabetes-endpoint-$env:ENVIRONMENT"
$DEPLOYMENT_NAME = "diabetes-deployment-$env:ENVIRONMENT"
$MODEL_NAME = "diabetes-model-prod"

Write-Host "Deploying model to endpoint: $ENDPOINT_NAME"

# Determine model version to use
if ([string]::IsNullOrEmpty($env:MODEL_VERSION)) {
    Write-Host "No specific version provided. Getting latest version..."
    
    # Get latest model version
    $MODEL_VERSION = az ml model list `
      --resource-group $env:RESOURCE_GROUP `
      --workspace-name $env:WORKSPACE_NAME `
      --name $MODEL_NAME `
      --query "[0].version" -o tsv
    
    Write-Host "Latest model version: $MODEL_VERSION"
} else {
    $MODEL_VERSION = $env:MODEL_VERSION
    Write-Host "Using specified model version: $MODEL_VERSION"
}

# Read deployment template
$deploymentContent = Get-Content -Path "deploy/deployment.yml" -Raw

# Replace placeholders
$deploymentContent = $deploymentContent -replace "{{ENDPOINT_NAME}}", $ENDPOINT_NAME
$deploymentContent = $deploymentContent -replace "{{DEPLOYMENT_NAME}}", $DEPLOYMENT_NAME
$deploymentContent = $deploymentContent -replace "{{MODEL_NAME}}", $MODEL_NAME
$deploymentContent = $deploymentContent -replace "{{MODEL_VERSION}}", $MODEL_VERSION

# Save updated deployment file
$deploymentContent | Set-Content -Path "deploy/deployment-temp.yml"

# Create or update deployment
Write-Host "Creating deployment with model ${MODEL_NAME}:${MODEL_VERSION}..."
az ml online-deployment create `
  --resource-group $env:RESOURCE_GROUP `
  --workspace-name $env:WORKSPACE_NAME `
  --file deploy/deployment-temp.yml `
  --all-traffic

Write-Host "Model deployed successfully to $ENDPOINT_NAME"

# Save deployment info
"DEPLOYMENT_NAME=$DEPLOYMENT_NAME" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
"MODEL_VERSION=$MODEL_VERSION" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append

# Clean up temp file
Remove-Item "deploy/deployment-temp.yml" -ErrorAction SilentlyContinue