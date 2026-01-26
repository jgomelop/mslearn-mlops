"""
Script to deploy an already-registered X-ray classification model to Azure ML.
Use this when your model is already registered in Azure ML Models section.
"""

import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Environment,
    CodeConfiguration,
)
from azure.identity import InteractiveBrowserCredential


def resolve_environment(ml_client, args):
    """
    Resolve which Environment to use for deployment:
    - If args.environment is provided, use an existing env (string or fetched object).
    - Else, build an inline environment with a safe, compatible stack.
    """
    # Priority: explicit existing environment
    if args.environment:
        print("\n3. Using existing Azure ML Environment...")
        env_ref = args.environment.strip()

        # If user passed the full shorthand "azureml:NAME:VERSION"
        if env_ref.startswith("azureml:"):
            print(f"✓ Using environment reference: {env_ref}")
            return env_ref  # string reference is accepted by deployment

        # Else allow "NAME:VERSION" or "NAME"
        parts = env_ref.split(":")
        if len(parts) == 2:
            name, version = parts
            env_obj = ml_client.environments.get(name=name, version=version)
            print(f"✓ Resolved environment: {env_obj.name}:{env_obj.version}")
            return env_obj
        elif len(parts) == 1:
            name = parts[0]
            try:
                env_obj = ml_client.environments.get(name=name, label="latest")
                print(f"✓ Resolved environment by label 'latest': {env_obj.name}:{env_obj.version}")
                return env_obj
            except Exception as e:
                raise RuntimeError(
                    f"Could not find latest environment for name '{name}'. "
                    f"Provide an explicit version (NAME:VERSION) or ensure a 'latest' label exists. Details: {e}"
                )
        else:
            raise ValueError(
                "Invalid --environment format. Use 'azureml:NAME:VERSION', 'NAME:VERSION', or 'NAME'"
            )

    # Fallback: Build inline environment
    print("\n3. Creating inline environment (no --environment provided)...")
    if args.conda_file:
        environment = Environment(
            name="xray-classification-env",
            description="Environment for X-ray classification model",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file=args.conda_file,
        )
    else:
        # Safer defaults: ensure azureml-inference-server-http is present and Torch CPU is pinned
        environment = Environment(
            name="xray-classification-env",
            description="Inline env for X-ray classification (CPU inference)",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        )
        environment.conda_file = {
            "name": "xray-inference-inline",
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                "python=3.10",
                "pip",
                "numpy=1.26.4",
                "pandas",
                "scikit-learn",
                "pillow",
                "scikit-image",
                "imageio",
                "tqdm",
                "requests",
                "matplotlib",
                {
                    "pip": [
                        "--extra-index-url https://download.pytorch.org/whl/cpu",
                        "azureml-inference-server-http>=0.8.0",
                        "torch==2.2.2",
                        "torchvision==0.17.2",
                        "torchaudio==2.2.2",
                        "pydicom",
                        "torchxrayvision",
                    ]
                },
            ],
        }

    print("✓ Environment configured")
    return environment


def main(args):
    """Deploy the already-registered model to Azure ML"""

    # Connect to Azure ML workspace using interactive browser authentication
    print("Authenticating with Azure (browser window will open)...")
    credential = InteractiveBrowserCredential(tenant_id=args.tenant_id)

    print("Connecting to Azure ML workspace...")
    ml_client = MLClient(
        credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )
    print("✓ Connected successfully")

    # Get the registered model
    print(f"\n1. Getting registered model: {args.model_name}...")
    if args.model_version:
        model = ml_client.models.get(name=args.model_name, version=args.model_version)
    else:
        model = ml_client.models.get(name=args.model_name, label="latest")

    print(f"✓ Found model: {model.name}, Version: {model.version}")
    print(f"  Model path: {model.path}")

    # Create or update endpoint
    print(f"\n2. Creating endpoint: {args.endpoint_name}...")
    endpoint = ManagedOnlineEndpoint(
        name=args.endpoint_name,
        description="Endpoint for X-ray classification model",
        auth_mode="key",
    )

    try:
        endpoint_result = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"✓ Endpoint created: {endpoint_result.name}")
    except Exception as e:
        print(f"Endpoint may already exist: {e}")
        endpoint_result = ml_client.online_endpoints.get(args.endpoint_name)
        print(f"✓ Using existing endpoint: {endpoint_result.name}")

    # Resolve environment (existing or inline)
    environment = resolve_environment(ml_client, args)

    # Create deployment
    print(f"\n4. Creating deployment: {args.deployment_name}...")
    deployment = ManagedOnlineDeployment(
        name=args.deployment_name,
        endpoint_name=args.endpoint_name,
        model=model.id,
        environment=environment,  # can be Environment object or "azureml:NAME:VERSION"
        code_configuration=CodeConfiguration(
            code=args.code_path,
            scoring_script=args.scoring_script,
        ),
        instance_type=args.instance_type,
        instance_count=args.instance_count,
    )

    print("Deploying... This may take 10-15 minutes.")
    print("You can monitor progress in Azure ML Studio.")

    deployment_result = ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"✓ Deployment created: {deployment_result.name}")

    # Set deployment to receive 100% of traffic
    print("\n5. Routing traffic to deployment...")
    endpoint_result.traffic = {args.deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint_result).result()
    print("✓ Traffic routing updated (100% to this deployment)")

    # Get endpoint details
    endpoint_result = ml_client.online_endpoints.get(args.endpoint_name)
    keys = ml_client.online_endpoints.get_keys(args.endpoint_name)

    print("\n" + "="*70)
    print("DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Endpoint name: {endpoint_result.name}")
    print(f"Scoring URI: {endpoint_result.scoring_uri}")
    print(f"\nPrimary key: {keys.primary_key}")
    print(f"Secondary key: {keys.secondary_key}")
    print("="*70)
    print("\nTo test your endpoint, run:")
    print(f"python test_endpoint.py \\")
    print(f'  --endpoint_uri "{endpoint_result.scoring_uri}" \\')
    print(f'  --api_key "{keys.primary_key}" \\')
    print(f'  --image_path "path/to/xray.jpg"')
    print("="*70)

    return endpoint_result


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Deploy already-registered X-ray classification model"
    )

    # Azure ML workspace arguments
    parser.add_argument("--subscription_id", type=str, required=True)
    parser.add_argument("--resource_group", type=str, required=True)
    parser.add_argument("--workspace_name", type=str, required=True)
    parser.add_argument("--tenant_id", type=str, required=True)

    # Model arguments
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_version", type=str, default=None)

    # Endpoint arguments
    parser.add_argument("--endpoint_name", type=str, required=True)
    parser.add_argument("--deployment_name", type=str, default="blue")

    # Code configuration
    parser.add_argument("--code_path", type=str, default="./src/model")
    parser.add_argument("--scoring_script", type=str, default="score.py")

    # Environment selection
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Existing Azure ML environment to use. "
             "Accepted: 'azureml:NAME:VERSION', 'NAME:VERSION', or 'NAME' (uses label 'latest'). "
             "If omitted, an inline env is created."
    )
    parser.add_argument(
        "--conda_file",
        type=str,
        default=None,
        help="Path to a conda environment YAML (only used if --environment is not provided)"
    )

    # Infrastructure arguments
    parser.add_argument(
        "--instance_type",
        type=str,
        default="Standard_F2s_v2",  # cheaper default for testing
        help="VM instance type (default: Standard_F2s_v2)"
    )
    parser.add_argument("--instance_count", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
