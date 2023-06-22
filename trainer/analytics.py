import os

import requests

telemetry = os.environ.get("TRAINER_TELEMETRY")


def ping_training_run():
    if telemetry == "0":
        return
    URL = "https://coqui.gateway.scarf.sh/trainer/training_run"
    _ = requests.get(URL, timeout=5)
