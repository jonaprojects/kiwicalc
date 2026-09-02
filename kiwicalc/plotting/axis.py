"""Axis configuration helpers used by KiwiCalc graphs."""

from fractions import Fraction
from math import degrees, pi
from typing import Mapping, Optional, Sequence, Tuple, Union

from matplotlib.ticker import AutoLocator, AutoMinorLocator, EngFormatter, FuncFormatter, MultipleLocator, ScalarFormatter


TickMode = Optional[str]
UnitsInput = Optional[Union[str, Sequence[Optional[str]], Mapping[str, Optional[str]]]]


def axis_label(label: Optional[str], unit: Optional[str]) -> str:
    label = "" if label is None else str(label)
    if not unit:
        return label
    if label.rstrip().endswith(f"({unit})"):
        return label
    return f"{label} ({unit})" if label else str(unit)


def normalize_units(units: UnitsInput, names: Sequence[str]) -> Tuple[Optional[str], ...]:
    if units is None:
        return (None,) * len(names)
    if isinstance(units, str):
        return (units,) * len(names)
    if isinstance(units, Mapping):
        return tuple(units.get(name) for name in names)
    values = tuple(units)
    if len(values) != len(names):
        joined = ", ".join(names)
        raise ValueError(f"units must provide one value for each axis: {joined}")
    return values


def _pi_label(value: float, _position=None) -> str:
    ratio = value / pi
    fraction = Fraction(ratio).limit_denominator(12)
    if abs(float(fraction) - ratio) > 1e-8:
        return f"{value:g}"
    numerator, denominator = fraction.numerator, fraction.denominator
    if numerator == 0:
        return "0"
    sign = "-" if numerator < 0 else ""
    magnitude = abs(numerator)
    coefficient = "" if magnitude == 1 else str(magnitude)
    if denominator == 1:
        return rf"${sign}{coefficient}\pi$"
    return rf"${sign}\frac{{{coefficient}\pi}}{{{denominator}}}$"


def _degree_label(value: float, _position=None) -> str:
    angle = degrees(value)
    rounded = round(angle)
    shown = str(rounded) if abs(angle - rounded) < 1e-8 else f"{angle:g}"
    return f"{shown}°"


def configure_ticks(axis, mode: TickMode, *, unit: Optional[str] = None, pi_step: float = pi / 2,
                    degree_step: float = pi / 4, power_limits=(-3, 4)) -> None:
    """Configure one Matplotlib Axis with a friendly named tick mode."""
    if mode is None:
        return
    if not isinstance(mode, str):
        raise TypeError("tick mode must be 'plain', 'pi', 'degrees', 'scientific', or 'engineering'")
    normalized = mode.strip().lower().replace("_", "-")
    if normalized in ("plain", "decimal", "auto"):
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
    elif normalized in ("pi", "π", "radians"):
        if pi_step <= 0:
            raise ValueError("pi_step must be positive")
        axis.set_major_locator(MultipleLocator(pi_step))
        axis.set_major_formatter(FuncFormatter(_pi_label))
    elif normalized in ("degrees", "degree", "deg"):
        if degree_step <= 0:
            raise ValueError("degree_step must be positive")
        axis.set_major_locator(MultipleLocator(degree_step))
        axis.set_major_formatter(FuncFormatter(_degree_label))
    elif normalized in ("scientific", "sci"):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits(tuple(power_limits))
        axis.set_major_formatter(formatter)
    elif normalized in ("engineering", "eng"):
        axis.set_major_formatter(EngFormatter(unit=unit or "", sep=" "))
    else:
        raise ValueError("tick mode must be 'plain', 'pi', 'degrees', 'scientific', or 'engineering'")


def configure_minor_ticks(axis, enabled: bool, subdivisions: int = 2) -> None:
    if enabled:
        if subdivisions < 2:
            raise ValueError("minor tick subdivisions must be at least 2")
        axis.set_minor_locator(AutoMinorLocator(subdivisions))
    else:
        from matplotlib.ticker import NullLocator
        axis.set_minor_locator(NullLocator())


def set_axes_at_origin(ax, enabled: bool) -> None:
    """Move 2D axes to the origin, or restore the normal rectangular frame."""
    # Keep descriptive labels outside the data region even when their spines
    # cross through zero.
    ax.xaxis.set_label_coords(0.5, -0.075)
    ax.yaxis.set_label_coords(-0.075, 0.5)
    if enabled:
        ax.spines["bottom"].set_position("zero")
        ax.spines["left"].set_position("zero")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.spines["bottom"].set_position(("outward", 0))
        ax.spines["left"].set_position(("outward", 0))
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)


__all__ = ["TickMode", "UnitsInput"]
