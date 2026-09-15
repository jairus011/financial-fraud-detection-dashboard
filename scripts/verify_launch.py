"""Start a real Streamlit server, check HTTP health, and stop the test server."""
import http.client
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
with tempfile.TemporaryFile(mode="w+") as log:
    process = subprocess.Popen([sys.executable, "scripts/start_dashboard.py"], cwd=ROOT,
                                env={**os.environ, "PORT": str(port)},
                                stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 40
        healthy = False
        while time.monotonic() < deadline and process.poll() is None:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                connection.request("GET", "/_stcore/health")
                response = connection.getresponse()
                healthy = response.status == 200 and response.read() == b"ok"
                if healthy:
                    break
            except OSError:
                pass
            finally:
                connection.close()
            time.sleep(.25)
        if not healthy:
            log.seek(0)
            raise RuntimeError("Streamlit did not become healthy.\n" + log.read())
        print("PASS: Hosted startup respected PORT and /_stcore/health returned HTTP 200 / ok.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
