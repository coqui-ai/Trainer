import requests


def ping_training_run():
    URL = "https://coqui.gateway.scarf.sh/trainer/training_run"
    _ = requests.get(URL)