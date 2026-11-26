# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""Drives a headless Chrome over the DevTools protocol: load a page, evaluate a JavaScript expression, take a screenshot. Used to draw the frames of the animation and to preview single frames.

The browser is found without any set-up. ``chrome_path`` takes, in this order: the file that the environment variable ``CYPMODE_CHROME`` names; a Chrome or Chromium that is installed on the machine;
a Chrome for Testing that an earlier run has put into the cache (``~/.cache/cypmode``, or the folder of ``CYPMODE_CACHE``); and otherwise it downloads Chrome for Testing, Google's own channel of browsers that
programs drive, once, checks the size and the SHA-256 of the file against the values that are written below, and unpacks it into the cache. ``CYPMODE_NO_DOWNLOAD=1`` forbids the download."""

import base64
import functools
import hashlib
import json
import os
import platform
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Self

from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

WIDTH, HEIGHT = 1920, 1080
CHROME_FOR_TESTING = "158.0.8127.88"  # the stable channel on 2026-11-26: https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json
CACHE = Path(os.environ.get("CYPMODE_CACHE") or Path.home() / ".cache" / "cypmode") / "chrome-for-testing"
BUILDS = {  # platform: (size in bytes, SHA-256 of the zip, the browser inside it); computed on 2026-11-26 from the files of Google's bucket chrome-for-testing-public, whose own MD5 each one matched
    "mac-arm64": (200337306, "6caf16b007aafaf6d4a4db252262be8ebbf0d91a2af005257e388996dbe463d6", "chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"),
    "mac-x64": (213548436, "4529a2bdbea10cf69f79b8b0ff43a5a23e8dc8b033da85cbe873afb81855579a", "chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"),
    "linux64": (208082347, "59e3b43f8b3aaa292238fbdcfb36a93127debd53e212510dcdc2cc8812b4b57e", "chrome-linux64/chrome"),
    "linux-arm64": (205794810, "91669b5de79af808d498037dd974e4b09461776a1b5240b3bb656abd17e5d3ab", "chrome-linux-arm64/chrome"),
    "win64": (214931627, "fda73ca9ffc7f02c5c85165cb472152540a76a7cecd8ec11314ea781e4454109", "chrome-win64/chrome.exe"),
}


class ChromeUnavailable(RuntimeError):
    """No Chrome is on the machine and none could be downloaded; the message says what to do."""


def build_name() -> str | None:
    """The name of the Chrome for Testing build that fits this machine, or None."""
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Darwin":
        return "mac-arm64" if arm else "mac-x64"
    if system == "Linux":
        return "linux-arm64" if arm else "linux64" if machine in ("x86_64", "amd64") else None
    if system == "Windows":
        return "win64" if machine in ("amd64", "x86_64") else None
    return None


def installed_chrome() -> str | None:
    """A Chrome or Chromium that is installed on the machine, the first that is found."""
    system, candidates = platform.system(), []
    if system == "Darwin":
        for base in (Path("/Applications"), Path.home() / "Applications"):
            candidates += [base / "Google Chrome.app/Contents/MacOS/Google Chrome", base / "Chromium.app/Contents/MacOS/Chromium"]
    elif system == "Windows":
        candidates = [Path(os.environ[variable]) / "Google/Chrome/Application/chrome.exe" for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA") if os.environ.get(variable)]
    else:
        candidates = [Path(found) for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome") if (found := shutil.which(name))]
    return next((str(path) for path in candidates if path.exists()), None)


def _extract(archive: Path, target: Path) -> None:
    """Unpack a zip with its file modes and its symbolic links, which the app of macOS needs and ``extractall`` does not restore."""
    root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            destination = target / info.filename
            if not destination.resolve().is_relative_to(root):
                raise ChromeUnavailable(f"the archive names a file outside its folder: {info.filename}")
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                os.symlink(bundle.read(info).decode(), destination)
                continue
            with bundle.open(info) as source, open(destination, "wb") as out:
                shutil.copyfileobj(source, out)
            if mode:
                os.chmod(destination, stat.S_IMODE(mode) | stat.S_IRUSR | stat.S_IWUSR)


def _download(name: str) -> None:
    """Fetch Chrome for Testing for ``name`` into the cache. The file is checked before it is unpacked, and the folder appears in the cache only when it is complete."""
    size, sha256, _ = BUILDS[name]
    target = CACHE / CHROME_FOR_TESTING / name
    url = f"https://storage.googleapis.com/chrome-for-testing-public/{CHROME_FOR_TESTING}/{name}/chrome-{name}.zip"
    print(f"No Chrome was found: downloading Chrome for Testing {CHROME_FOR_TESTING} ({name}, {size / 1e6:.0f} MB) once into {target}. "
          "Set CYPMODE_CHROME to a browser of your own, or CYPMODE_NO_DOWNLOAD=1 to forbid the download.", file=sys.stderr, flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="download-", dir=target.parent))
    try:
        archive, digest, received = work / "chrome.zip", hashlib.sha256(), 0
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "cypmode-chrome-fetch"}), timeout=60) as response, open(archive, "wb") as out:
            for block in iter(lambda: response.read(1 << 20), b""):
                out.write(block)
                digest.update(block)
                received += len(block)
        if received != size or digest.hexdigest() != sha256:
            raise ChromeUnavailable(f"the file from {url} has {received} bytes and SHA-256 {digest.hexdigest()}; expected {size} bytes and {sha256}, so it was not used")
        _extract(archive, work / "unpacked")
        try:
            (work / "unpacked").replace(target)
        except OSError:  # another run put the same folder there first
            if not (target / BUILDS[name][2]).exists():
                raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


@functools.cache
def chrome_path() -> str:
    """The browser to start: the file of ``CYPMODE_CHROME``, an installed Chrome, the Chrome for Testing of the cache, or one that is downloaded now. Raises ``ChromeUnavailable`` when none of them is possible."""
    given = os.environ.get("CYPMODE_CHROME")
    if given:
        if Path(given).exists():
            return given
        raise ChromeUnavailable(f"CYPMODE_CHROME names {given}, which does not exist")
    found = installed_chrome()
    if found:
        return found
    name = build_name()
    if name is None:
        raise ChromeUnavailable(f"no Chrome is installed, and Chrome for Testing has no build for {platform.system()} {platform.machine()}; set CYPMODE_CHROME to a Chrome or Chromium")
    cached = CACHE / CHROME_FOR_TESTING / name / BUILDS[name][2]
    if cached.exists():
        return str(cached)
    if os.environ.get("CYPMODE_NO_DOWNLOAD", "0") not in ("", "0"):
        raise ChromeUnavailable("no Chrome is installed, and the download of Chrome for Testing is switched off by CYPMODE_NO_DOWNLOAD; install Google Chrome or set CYPMODE_CHROME")
    try:
        _download(name)
    except (OSError, zipfile.BadZipFile) as error:
        raise ChromeUnavailable(f"no Chrome is installed, and the download of Chrome for Testing failed ({error}); install Google Chrome or set CYPMODE_CHROME") from error
    return str(cached)


@functools.cache
def unusable_reason() -> str | None:
    """None when a browser can be started here and opens a local debugging port; otherwise the reason, in a sentence that a test can print when it skips itself.
    A browser that starts and then gives a wrong answer is not covered: that is a failure of the code under test and stays one."""
    try:
        path = chrome_path()
    except ChromeUnavailable as error:
        return str(error)
    try:
        with Chrome():
            return None
    except (RuntimeError, OSError) as error:
        return f"Chrome ({path}) is on this machine but cannot open a local debugging port here ({error}); the tests that draw a page need it"


def free_port() -> int:
    """A port that no process listens on at the moment."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Chrome:
    def __init__(self, port: int | None = None):
        self.port = port  # None: a free port is taken for every start, so that no other browser on the machine shares it
        self._active_port = port
        self._id = 0
        self._profile = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # a helper process of the browser may still write into it while it is removed
        self._process = None
        self._socket = None
        self._page = None
        self.restarts = 0

    def __enter__(self) -> Self:
        self._launch()
        return self

    def _launch(self) -> None:
        self._active_port = self.port or free_port()
        flags = ["--headless=new", f"--remote-debugging-port={self._active_port}", f"--user-data-dir={self._profile.name}", "--hide-scrollbars", f"--window-size={WIDTH},{HEIGHT}",
                 "--force-device-scale-factor=1", "--allow-file-access-from-files", "--disable-background-timer-throttling", "about:blank"]
        if hasattr(os, "geteuid") and os.geteuid() == 0:  # the sandbox of Chrome refuses to start under root, which is how a container runs
            flags.insert(0, "--no-sandbox")
        self._process = subprocess.Popen([chrome_path(), *flags], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self._active_port}/json"))
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
            self._profile = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
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
        """Close the connection, end the process and remove the profile of the browser that is running, if any.
        A browser that does not end within ten seconds (a frozen or a loaded one) is killed, so that no step of the end can raise."""
        try:
            self._socket.close()
        except (OSError, AttributeError, ConnectionClosed):
            pass
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
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
        self._stop()

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
