# SPDX-FileCopyrightText: 2026 Muhammad Talaat
# SPDX-License-Identifier: AGPL-3.0-or-later
# Additional terms under section 7(b) of the licence: see NOTICE.

"""How the simulators of a part are registered. A part module holds ``SIMS = {"1.3": (controls, view), ...}``: ``controls(mo)`` returns the interface elements of the section as a dictionary and
``view(mo, values)`` the page of the section from the values of those elements. An entry may have a third member, ``embed(mo, values)``, for an element that marimo does not show inside content that is made on demand (a frame that holds an HTML page, such as the viewer of a structure): it is made at once and shown between the controls and the page. The app makes one dictionary of elements per part and renders a section only when it is opened, so that a control of one
section never recomputes the others. Each section shows its own elements above its page."""


def controls(mo, sims: dict):
    return mo.ui.dictionary({number: mo.ui.dictionary(entry[0](mo)) for number, entry in sims.items()})


def views(mo, ctl, sims: dict) -> dict:
    return {number: mo.vstack([mo.vstack([ctl[number][name] for name in ctl[number].elements]), *([entry[2](mo, ctl.value[number])] if len(entry) > 2 else []), mo.lazy(lambda number=number, view=entry[1]: view(mo, ctl.value[number]))]) for number, entry in sims.items()}
