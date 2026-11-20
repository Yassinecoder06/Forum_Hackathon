"""Private connection to the optional host-side motion inference worker."""
import http.client
import json
import os
from pathlib import Path
import socket


def available():
    path = os.getenv("SIGNACULTURE_INFERENCE_SOCKET")
    return bool(path and Path(path).is_socket())


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(os.environ["SIGNACULTURE_INFERENCE_SOCKET"])


def infer(path, motion, timeout):
    connection = UnixConnection("localhost", timeout=timeout)
    try:
        connection.request("POST", path, json.dumps(motion, allow_nan=False), {"Content-Type": "application/json"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"Inference worker returned HTTP {response.status}")
        return json.load(response)
    finally:
        connection.close()
