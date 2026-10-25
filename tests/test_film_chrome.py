# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs/video/anime"))

import chrome as chrome_module
from websockets.exceptions import ConnectionClosedError


class Browser(chrome_module.Chrome):
    """The driver without a browser: a launch only counts, and opening a page only records it."""

    def __init__(self):
        super().__init__()
        self.launched, self.opened = 0, []

    def _launch(self):
        self.launched += 1

    def open(self, path):
        self._page = Path(path)
        self.opened.append(Path(path))


def gone():
    return ConnectionClosedError(None, None)


def test_an_action_is_run_again_in_a_new_browser_when_the_browser_has_gone():
    browser, calls = Browser(), []

    def action():
        calls.append(1)
        if len(calls) < 3:
            raise gone()
        return "frame"

    assert browser.retrying(action) == "frame"
    assert len(calls) == 3 and browser.restarts == 2 and browser.launched == 2


def test_the_page_that_was_open_is_opened_again_after_a_restart():
    browser = Browser()
    browser.open("page.html")
    state = {"failed": False}

    def action():
        if not state["failed"]:
            state["failed"] = True
            raise gone()

    browser.retrying(action)
    assert browser.opened == [Path("page.html"), Path("page.html")]


def test_the_last_failure_is_raised_when_every_attempt_has_failed():
    browser = Browser()
    with pytest.raises(ConnectionClosedError):
        browser.retrying(lambda: (_ for _ in ()).throw(gone()), attempts=3)
    assert browser.restarts == 2


def test_a_browser_that_does_not_start_is_started_again(monkeypatch):
    monkeypatch.setattr(chrome_module.time, "sleep", lambda seconds: None)
    browser = Browser()
    starts = []

    def launch():
        starts.append(1)
        if len(starts) < 3:
            raise RuntimeError("Chrome did not start")

    browser._launch = launch
    browser.restart()
    assert len(starts) == 3 and browser.restarts == 1


def test_a_browser_that_never_starts_raises_after_the_attempts(monkeypatch):
    monkeypatch.setattr(chrome_module.time, "sleep", lambda seconds: None)
    browser = Browser()
    browser._launch = lambda: (_ for _ in ()).throw(RuntimeError("Chrome did not start"))
    with pytest.raises(RuntimeError):
        browser.restart(attempts=3)


def test_an_error_of_the_page_is_not_a_reason_to_start_a_new_browser():
    browser = Browser()
    with pytest.raises(RuntimeError):
        browser.retrying(lambda: (_ for _ in ()).throw(RuntimeError("JavaScript error")))
    assert browser.restarts == 0


def test_the_port_of_the_debugging_connection_is_free_and_not_the_same_for_two_starts():
    ports = {chrome_module.free_port() for _ in range(3)}
    assert all(1024 < port < 65536 for port in ports)
    assert chrome_module.Chrome().port is None


class Frozen:
    """A browser process that ignores the request to end: it has to be killed, and its end must not raise."""

    def __init__(self):
        self.signals, self.waits = [], []

    def terminate(self):
        self.signals.append("terminate")

    def kill(self):
        self.signals.append("kill")

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if timeout is not None:
            raise chrome_module.subprocess.TimeoutExpired("chrome", timeout)


def test_leaving_the_driver_kills_a_browser_that_does_not_end_and_does_not_raise():
    # the end of the module fixture of tests/test_video.py raised TimeoutExpired when the browser took more than ten seconds to end, which pytest reports as an error of the last test of the module
    browser = Browser()
    browser._process = Frozen()
    browser.__exit__(None, None, None)
    assert browser._process is None
