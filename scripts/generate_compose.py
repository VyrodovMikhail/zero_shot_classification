"""
Generates docker-compose.<server>.yml for every host defined in cluster.json.

Usage:
    python generate_compose.py --cluster cluster.json \
                               --images  images.yaml \
                               --base-port 8100
Afterwards copy the corresponding compose file to the target server
and run: docker compose -f docker-compose.<server>.yml up -d
"""
import argparse
import json
import re
from pathlib import Path

import yaml

GPU_RE = re.compile(r"cuda:(\d+)(?:-(\d+))?$")


def expand(spec: str):
    m = GPU_RE.fullmatch(spec)
    if not m:
        raise ValueError(f"Invalid GPU format: {spec}")
    a, b = int(m.group(1)), int(m.group(2) or m.group(1))
    return list(range(a, b + 1))


def make_service(model, image, gpu, external_port, otel_endpoint=None, otel_protocol="grpc"):
    """
    Returns a YAML snippet describing the service.
    • Inside the container FastAPI listens on 8000/tcp,
      external_port is exposed on the host.
    """
    svc_name = f"{model.split('/')[-1].replace('.', '-')}-gpu{gpu}"
    env = [f"MODEL_ID={model}", "HUGGINGFACE_HUB_TOKEN=/run/secrets/hf_token"]
    if otel_endpoint:
        env.extend(
            [
                f"OTEL_EXPORTER_OTLP_ENDPOINT={otel_endpoint}",
                f"OTEL_EXPORTER_OTLP_PROTOCOL={otel_protocol}",
                f"OTEL_SERVICE_NAME={svc_name}",
                "OTEL_TRACES_EXPORTER=otlp",
                "OTEL_METRICS_EXPORTER=otlp",
            ]
        )

    return svc_name, {
        "image": image,
        "container_name": svc_name,
        "restart": "unless-stopped",
        "ports": [f"{external_port}:8000"],
        "deploy": {
            "resources": {
                "reservations": {
                    "devices": [
                        {
                            "driver": "nvidia",
                            "device_ids": [str(gpu)],
                            "capabilities": ["gpu"],
                        }
                    ]
                }
            }
        },
        "environment": env,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True, help="cluster.json")
    ap.add_argument("--images", required=True, help="images.yaml")
    ap.add_argument("--base-port", type=int, default=8100, help="first external port")
    ap.add_argument("--otel-endpoint", help="OTLP collector endpoint")
    ap.add_argument("--otel-protocol", default="grpc", help="OTLP protocol")
    args = ap.parse_args()

    cluster = json.loads(Path(args.cluster).read_text())
    images = yaml.safe_load(Path(args.images).read_text())

    for server, devices in cluster.items():
        compose = {"version": "3.9", "services": {}}
        compose["secrets"] = {"hf_token": {"file": "./hf_token.txt"}}
        port = args.base_port
        for gpu_spec, model in devices.items():
            image = images.get(model)
            if image is None:
                print(f"No docker image found for model {model}")
                continue
            for gpu in expand(gpu_spec):
                svc_name, svc_def = make_service(
                    model,
                    image,
                    gpu,
                    port,
                    args.otel_endpoint,
                    args.otel_protocol,
                )
                compose["services"][svc_name] = svc_def
                port += 1

        out = Path(f"docker-compose.{server}.yml")
        out.write_text(yaml.dump(compose, sort_keys=False, allow_unicode=True))
        print(f"Generated {out} file")


if __name__ == "__main__":
    main()
