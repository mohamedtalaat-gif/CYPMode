"""The catalog of figures: every figure is registered with its title, caption, data files and the notebook section that uses it.

``build_all.py`` runs the registered functions and fails when a catalogued figure is not produced; ``CATALOG.md`` is generated from here.
"""

from dataclasses import dataclass, field


@dataclass
class Figure:
    figure_id: str
    group: str
    title: str
    caption: object  # str, or a callable returning str when the text quotes numbers read from the data files
    make: object  # make(theme_name) -> matplotlib Figure
    data: tuple[str, ...] = ()
    section: str = ""
    animation: object | None = None  # callable(theme_name) -> Path of a gif, for dynamic processes
    files: list = field(default_factory=list)


REGISTRY: dict[str, Figure] = {}


def figure(figure_id: str, group: str, title: str, caption, data: tuple[str, ...] = (), section: str = "", animation=None):
    def decorator(make):
        if figure_id in REGISTRY:
            raise ValueError(f"duplicate figure id {figure_id}")
        REGISTRY[figure_id] = Figure(figure_id, group, title, caption, make, data, section, animation)
        return make

    return decorator


def caption_text(fig: Figure) -> str:
    return fig.caption() if callable(fig.caption) else fig.caption
