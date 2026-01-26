"""
Submit job using Interactive Browser Authentication (no Azure CLI needed)
Uses existing environment and compute cluster
"""

import os
import json
from azure.ai.ml import MLClient, command, Input
from azure.identity import InteractiveBrowserCredential
from azure.ai.ml.constants import AssetTypes

# ============================================
# CONFIGURATION - Edit these values
# ============================================
SUBSCRIPTION_ID = "af5c874b-dab2-4b07-a93e-afab2d1a99ec"
RESOURCE_GROUP = "ml-resource-group"
WORKSPACE_NAME = "azml-workspace"

# CODE PATH - Point to directory containing train.py
CODE_PATH = "src/xray_model"

# COMPUTE - Use existing compute cluster name
COMPUTE_NAME = "training-cpu-2"

# ENVIRONMENT - Use existing registered environment
EXISTING_ENVIRONMENT_NAME = "xray-training-cpu"
EXISTING_ENVIRONMENT_VERSION = "4"  # or "latest"

# DATA PATH
DATA_PATH = "azureml:xray_data_source:1"

# TRAINING PARAMETERS
MODEL_NAME = "densenet121-res224-all"
BATCH_SIZE = 2
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
EXPERIMENT_NAME = "xray-classification"

# ============================================
# JOB SUBMISSION
# ============================================

print("=" * 60)
print("Authenticating with Interactive Browser...")
print("=" * 60)
print("🌐 A browser window will open for authentication.")
print("Please sign in with your Azure account.")
print()

# Use Interactive Browser Credential
credential = InteractiveBrowserCredential()

# Connect to workspace
print("✓ Connecting to workspace...")
ml_client = MLClient(
    credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

print(f"✓ Connected to workspace: {WORKSPACE_NAME}")

# Configure environment - use existing registered environment
if EXISTING_ENVIRONMENT_VERSION.lower() == "latest":
    env_reference = f"azureml:{EXISTING_ENVIRONMENT_NAME}@latest"
else:
    env_reference = f"azureml:{EXISTING_ENVIRONMENT_NAME}:{EXISTING_ENVIRONMENT_VERSION}"

print(f"✓ Using existing environment: {env_reference}")

# Define job
print(f"✓ Configuring job...")
job = command(
    display_name=f"xray-training-{MODEL_NAME}-{NUM_EPOCHS}epochs",
    experiment_name=EXPERIMENT_NAME,
    code=CODE_PATH,
    command="python train.py "
            "--training_data ${{inputs.training_data}} "
            "--model_name ${{inputs.model_name}} "
            "--batch_size ${{inputs.batch_size}} "
            "--num_epochs ${{inputs.num_epochs}} "
            "--learning_rate ${{inputs.learning_rate}}",
    inputs={
        "training_data": Input(type=AssetTypes.URI_FOLDER, path=DATA_PATH),
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
    },
    environment=env_reference,
    compute=COMPUTE_NAME,
)

# Submit job
print(f"\n🚀 Submitting job...")
print(f"  - Compute: {COMPUTE_NAME}")
print(f"  - Environment: {EXISTING_ENVIRONMENT_NAME}:{EXISTING_ENVIRONMENT_VERSION}")
print(f"  - Experiment: {EXPERIMENT_NAME}")
print(f"  - Code Path: {CODE_PATH}")

returned_job = ml_client.jobs.create_or_update(job)

print("\n" + "=" * 60)
print("✅ JOB SUBMITTED SUCCESSFULLY!")
print("=" * 60)
print(f"Job Name: {returned_job.name}")
print(f"Job ID: {returned_job.id}")
print(f"Status: {returned_job.status}")
print(f"Experiment: {returned_job.experiment_name}")
print(f"Compute: {returned_job.compute}")
print(f"\n🔗 Studio URL:\n{returned_job.studio_url}")

# Print detailed job information (like az ml job create output)
print("\n" + "=" * 60)
print("JOB DETAILS")
print("=" * 60)

# Print services (including MLflow tracking)
if hasattr(returned_job, 'services') and returned_job.services:
    print("\nServices:")
    for service_name, service_config in returned_job.services.items():
        print(f"  {service_name}:")
        if hasattr(service_config, '__dict__'):
            for key, value in service_config.__dict__.items():
                if not key.startswith('_'):
                    print(f"    {key}: {value}")
        else:
            print(f"    {service_config}")

# Print as JSON for complete details
print("\n" + "=" * 60)
print("COMPLETE JOB CONFIGURATION (JSON)")
print("=" * 60)

# Convert job to dict
job_dict = returned_job._to_dict()

# Pretty print JSON
print(json.dumps(job_dict, indent=2, default=str))
print("\n")