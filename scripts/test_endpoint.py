
import argparse
import requests
import json
import base64

def main(args):
    # Read image as bytes
    with open(args.image_path, "rb") as f:
        img_bytes = f.read()

    # Base64-encode image
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # IMPORTANT: Use the key expected by score.py → "image"
    payload = {"image": img_b64}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}"
    }

    print("Sending request...")
    response = requests.post(
        args.endpoint_uri,
        headers=headers,
        data=json.dumps(payload)
    )

    print("Status:", response.status_code)
    print("Response:", response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint_uri", required=True, help="Azure ML endpoint scoring URI")
    parser.add_argument("--api_key", required=True, help="Primary or Secondary key")
    parser.add_argument("--image_path", required=True, help="Path to image file")
    args = parser.parse_args()
    main(args)
