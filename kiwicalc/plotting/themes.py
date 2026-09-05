"""Scoped visualization themes for KiwiCalc graphs."""

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from matplotlib.colors import is_color_like


@dataclass(frozen=True)
class PlotTheme:
    """A reusable, graph-local visual theme.

    Themes are applied directly to one KiwiCalc figure and axes. They never
    modify Matplotlib's global ``rcParams``.
    """

    name: str = "custom"
    figure_facecolor: str = "white"
    axes_facecolor: str = "white"
    foreground: str = "#202124"
    grid_color: str = "#b8bec8"
    grid_alpha: float = 0.35
    minor_grid_alpha: float = 0.16
    font_size: float = 11
    title_size: float = 14
    label_size: float = 12
    line_width: float = 2
    marker_size: float = 7
    color_cycle: Tuple[str, ...] = (
        "#0072B2", "#D55E00", "#009E73", "#CC79A7",
        "#E69F00", "#56B4E9", "#F0E442", "#000000",
    )
    grid: bool = True
    minor_grid: bool = False

    def __post_init__(self) -> None:
        positive = ("font_size", "title_size", "label_size", "line_width", "marker_size")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in ("grid_alpha", "minor_grid_alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        colors = {
            "figure_facecolor": self.figure_facecolor,
            "axes_facecolor": self.axes_facecolor,
            "foreground": self.foreground,
            "grid_color": self.grid_color,
        }
        for name, color in colors.items():
            if not is_color_like(color):
                raise ValueError(f"{name} must be a Matplotlib color")
        if not self.color_cycle:
            raise ValueError("color_cycle must contain at least one color")
        if any(not is_color_like(color) for color in self.color_cycle):
            raise ValueError("color_cycle must contain only Matplotlib colors")

    def with_overrides(self, **overrides: Any) -> 'PlotTheme':
        """Return a modified copy without changing this theme."""
        unknown = set(overrides) - set(self.__dataclass_fields__)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown theme option(s): {names}")
        if "color_cycle" in overrides:
            overrides["color_cycle"] = tuple(overrides["color_cycle"])
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["color_cycle"] = list(self.color_cycle)
        return data

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> 'PlotTheme':
        values = dict(data)
        if "color_cycle" in values:
            values["color_cycle"] = tuple(values["color_cycle"])
        return PlotTheme(**values)


_COLORBLIND = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
)


THEMES = {
    "classroom": PlotTheme(
        name="classroom", font_size=13, title_size=17, label_size=14,
        line_width=2.6, marker_size=8, grid_alpha=0.42, minor_grid=True,
        color_cycle=_COLORBLIND,
    ),
    "projector": PlotTheme(
        name="projector", font_size=16, title_size=21, label_size=18,
        line_width=3.2, marker_size=10, grid_alpha=0.45, minor_grid=False,
        color_cycle=("#0057B8", "#C43E00", "#00875A", "#8B2F97", "#222222"),
    ),
    "publication": PlotTheme(
        name="publication", font_size=9.5, title_size=11, label_size=10,
        line_width=1.5, marker_size=5, grid_alpha=0.25, minor_grid=False,
        color_cycle=("#111111", "#555555", "#888888", "#BBBBBB"),
    ),
    "engineering": PlotTheme(
        name="engineering", axes_facecolor="#f7f9fb", font_size=11,
        title_size=14, label_size=12, line_width=1.9, marker_size=6,
        grid_color="#8090a0", grid_alpha=0.38, minor_grid_alpha=0.18,
        minor_grid=True,
        color_cycle=("#0057A8", "#D14900", "#008450", "#7A3E9D", "#5F6368"),
    ),
    "colorblind": PlotTheme(
        name="colorblind", line_width=2.2, marker_size=7,
        color_cycle=_COLORBLIND,
    ),
}


ThemeInput = Optional[Union[str, PlotTheme, Mapping[str, Any]]]


def available_themes() -> Tuple[str, ...]:
    """Return the built-in theme names in display order."""
    return tuple(THEMES)


def get_theme(theme: ThemeInput, **overrides: Any) -> Optional[PlotTheme]:
    """Resolve a theme name, object, or mapping into a ``PlotTheme``."""
    if theme is None:
        if overrides:
            return PlotTheme().with_overrides(**overrides)
        return None
    if isinstance(theme, str):
        key = theme.strip().lower()
        if key not in THEMES:
            choices = ", ".join(available_themes())
            raise ValueError(f"Unknown theme {theme!r}. Choose from: {choices}")
        resolved = THEMES[key]
    elif isinstance(theme, PlotTheme):
        resolved = theme
    elif isinstance(theme, Mapping):
        resolved = PlotTheme.from_dict(theme)
    else:
        raise TypeError("theme must be a theme name, PlotTheme, mapping, or None")
    return resolved.with_overrides(**overrides) if overrides else resolved


def apply_theme(fig, ax, theme: Optional[PlotTheme]) -> None:
    """Apply a resolved theme to one figure and axes only."""
    if theme is None:
        return
    fig.set_facecolor(theme.figure_facecolor)
    ax.set_facecolor(theme.axes_facecolor)
    ax.set_prop_cycle(color=theme.color_cycle)
    ax.title.set_color(theme.foreground)
    ax.title.set_fontsize(theme.title_size)
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_color(theme.foreground)
        label.set_fontsize(theme.label_size)
    if hasattr(ax, "zaxis"):
        ax.zaxis.label.set_color(theme.foreground)
        ax.zaxis.label.set_fontsize(theme.label_size)
    ax.tick_params(axis="both", which="both", colors=theme.foreground, labelsize=theme.font_size)
    for spine in ax.spines.values():
        spine.set_color(theme.foreground)
    if theme.grid:
        ax.grid(True, which="major", color=theme.grid_color, alpha=theme.grid_alpha)
    else:
        ax.grid(False, which="major")
    if theme.minor_grid:
        ax.grid(True, which="minor", color=theme.grid_color, alpha=theme.minor_grid_alpha)
    else:
        ax.grid(False, which="minor")


__all__ = ["PlotTheme", "THEMES", "ThemeInput", "available_themes", "get_theme"]
