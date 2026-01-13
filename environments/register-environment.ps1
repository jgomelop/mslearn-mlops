$ErrorActionPreference = "Stop"

# Set your values
$RESOURCE_GROUP = "ml-resource-group"
$WORKSPACE_NAME = "azml-workspace"

Write-Host "Registering custom environment..."

# Register the environment
az ml environment create `
  --resource-group $RESOURCE_GROUP `
  --workspace-name $WORKSPACE_NAME `
  --file diabetes-training-env.yml