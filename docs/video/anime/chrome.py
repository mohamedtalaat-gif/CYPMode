# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Drives a headless Chrome over the DevTools protocol: load a page, evaluate a JavaScript expression, take a screenshot. Used to draw the frames of the animation and to preview single frames."""

import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Self

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
WIDTH, HEIGHT = 1920, 1080


class Chrome:
    def __init__(self, port: int = 9333):
        self.port = port
        self._id = 0
        self._profile = tempfile.TemporaryDirectory()
        self._process = None
        self._socket = None
        self._page = None
        self.restarts = 0

    def __enter__(self) -> Self:
        self._launch()
        return self

    def _launch(self) -> None:
        flags = ["--headless=new", f"--remote-debugging-port={self.port}", f"--user-data-dir={self._profile.name}", "--hide-scrollbars", f"--window-size={WIDTH},{HEIGHT}",
                 "--force-device-scale-factor=1", "--allow-file-access-from-files", "--disable-background-timer-throttling", "about:blank"]
        self._process = subprocess.Popen([CHROME, *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json"))
                break
            except OSError:
                time.sleep(0.2)
        else:
            raise RuntimeError("Chrome did not start")
        page = next(t for t in tabs if t["type"] == "page")
        self._socket = connect(page["webSocketDebuggerUrl"], max_size=None)
        self.call("Page.enable")
        self.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False)

    def restart(self, attempts: int = 12) -> None:
        """Start a new browser after the first one has gone (the page that was open is opened again); every call of the film is a function of the time it is given, so no state is lost.
        A browser that is ended while it starts, or before the page has loaded, is started again, up to ``attempts`` times."""
        self.restarts += 1
        for attempt in range(attempts):
            self._stop()
            self._profile = tempfile.TemporaryDirectory()
            try:
                self._launch()
                if self._page is not None:
                    self.open(self._page)
                return
            except (RuntimeError, ConnectionClosed, OSError):
                if attempt == attempts - 1:
                    raise
                time.sleep(1)

    def _stop(self) -> None:
        """Close the connection, end the process and remove the profile of the browser that is running, if any."""
        try:
            self._socket.close()
        except (OSError, AttributeError):
            pass
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        self._profile.cleanup()

    def retrying(self, action, attempts: int = 30):
        """Run ``action`` and, when the browser has gone while it ran, start a new one and run it again."""
        for attempt in range(attempts):
            try:
                return action()
            except (ConnectionClosed, OSError):
                if attempt == attempts - 1:
                    raise
                self.restart()

    def __exit__(self, *exc) -> None:
        if self._socket is not None:
            self._socket.close()
        if self._process is not None:
            self._process.terminate()
            self._process.wait(timeout=10)
        self._profile.cleanup()

    def call(self, method: str, **params) -> dict:
        self._id += 1
        self._socket.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            message = json.loads(self._socket.recv())
            if message.get("id") == self._id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message["result"]

    def open(self, path: Path) -> None:
        self._page = Path(path)
        self.call("Page.navigate", url=Path(path).resolve().as_uri())
        for _ in range(100):
            if self.evaluate("document.readyState") == "complete":
                return
            time.sleep(0.1)
        raise RuntimeError(f"{path} did not finish loading")

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            raise RuntimeError(f"JavaScript error: {details.get('exception', {}).get('description') or details['text']}")
        return result["result"].get("value")

    def screenshot(self, path: Path) -> None:
        data = self.call("Page.captureScreenshot", format="png")["data"]
        Path(path).write_bytes(base64.b64decode(data))
