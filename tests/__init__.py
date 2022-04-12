import os


def run_cli(command):
    exit_status = os.system(command)
    assert exit_status == 0, f" [!] command `{command}` failed."
