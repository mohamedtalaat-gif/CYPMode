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


def test_an_error_of_the_page_is_not_a_reason_to_start_a_new_browser():
    browser = Browser()
    with pytest.raises(RuntimeError):
        browser.retrying(lambda: (_ for _ in ()).throw(RuntimeError("JavaScript error")))
    assert browser.restarts == 0
