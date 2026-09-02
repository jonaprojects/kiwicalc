import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import kiwicalc as kw
from kiwicalc.plotting.motion import sample_frame


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_parameterized_sampling_supports_strings_and_scalar_callables():
    x = np.linspace(-1, 1, 5)
    sampled_x, y = sample_frame("f(x,a)=a*x^2", 3, x, "a")
    assert sampled_x == pytest.approx(x)
    assert y == pytest.approx(3 * x ** 2)

    _, scalar_y = sample_frame(lambda x, frequency: math.sin(frequency * x), 2, x, "frequency")
    assert scalar_y == pytest.approx(np.sin(2 * x))


def test_parameterized_sampling_accepts_curve_factories():
    factory = lambda radius: kw.Ellipse(radius, 1, samples=17)
    x, y = sample_frame(factory, 2, np.linspace(-1, 1, 5), "radius")
    assert len(x) == len(y) == 17
    assert max(x) == pytest.approx(2)


def test_animation_is_friendly_chain_aware_and_updates_titles():
    graph = kw.Graph2D().theme("engineering").horizontal_line(0, label="axis")
    controller = graph.animate_parameter(
        "f(x,a)=a*sin(x)", [1, 2, 3], parameter="a", show=False,
        title="Amplitude = {a:.1f}", label="wave", legend=True,
        line_style={"color": "navy"}, start=-np.pi, stop=np.pi, samples=101,
    )
    assert isinstance(controller, kw.GraphAnimation)
    assert controller.frames == (1, 2, 3)
    assert graph.ax.get_title() == "Amplitude = 1.0"
    assert {line.get_label() for line in graph.ax.lines} >= {"axis", "wave"}
    assert len(plt.get_fignums()) == 1

    controller.animation._func(3)
    assert graph.ax.get_title() == "Amplitude = 3.0"
    assert max(controller.line.get_ydata()) == pytest.approx(3)
    assert graph.ax.get_ylim()[1] > 3


def test_animate_alias_callable_titles_and_controller_controls(monkeypatch):
    controller = kw.Graph2D().animate(
        lambda x, speed: speed * x, frames=(0, 1), parameter="speed",
        title=lambda value: f"speed={value}", show=False,
    )
    assert controller.ax.get_title() == "speed=0"
    assert controller.fig is controller.graph.fig

    paused, resumed = [], []
    monkeypatch.setattr(controller.animation, "pause", lambda: paused.append(True))
    monkeypatch.setattr(controller.animation, "resume", lambda: resumed.append(True))
    assert controller.pause() is controller
    assert controller.resume() is controller
    assert paused == [True] and resumed == [True]

    with pytest.raises(ValueError, match="js.*video"):
        controller.to_html(mode="paper")


def test_animation_save_delegates_and_returns_the_output_path(tmp_path, monkeypatch):
    controller = kw.Graph2D().animate(lambda x, a: a * x, [1, 2], show=False)
    received = {}
    monkeypatch.setattr(controller.animation, "save", lambda path, **options: received.update(path=path, options=options))
    output = controller.save(tmp_path / "nested" / "motion.gif", fps=12, writer="pillow")
    assert output == Path(tmp_path / "nested" / "motion.gif")
    assert received["path"] == str(output)
    assert received["options"] == {"fps": 12, "writer": "pillow"}


def test_slider_interaction_updates_data_title_and_value():
    graph = kw.Graph2D().theme("classroom")
    control = graph.interactive_parameter(
        "f(x,k)=k*x", (0, 3), parameter="k", initial=1, step=0.25,
        show=False, title="k = {k:.2f}", line_style={"color": "crimson"},
        start=-2, stop=2,
    )
    assert isinstance(control, kw.GraphInteraction)
    assert control.value == pytest.approx(1)
    assert len(graph.fig.axes) == 2
    assert graph.ax.get_ylim() == pytest.approx((-6.96, 6.96))

    assert control.set_value(2.5) is control
    assert control.value == pytest.approx(2.5)
    assert control.ax.get_title() == "k = 2.50"
    assert control.line.get_ydata()[-1] == pytest.approx(5)
    assert control.disconnect() is control
    assert control._callback_id is None


def test_interact_alias_defaults_to_midpoint_and_supports_custom_values():
    control = kw.Graph2D().interact(
        lambda x, a: np.cos(a * x), (2, 4), parameter="a",
        values=[-1, 0, 1], show=False,
    )
    assert control.value == pytest.approx(3)
    assert control.line.get_xdata() == pytest.approx([-1, 0, 1])


def test_clear_stops_motion_controllers_and_removes_slider_axes(monkeypatch):
    graph = kw.Graph2D()
    animation = graph.animate(lambda x, a: a * x, [1, 2], show=False)
    interaction = graph.interact(lambda x, a: a * x, (0, 2), show=False)
    paused = []
    monkeypatch.setattr(animation, "pause", lambda: paused.append(True))
    assert len(graph.fig.axes) == 2

    graph.clear()
    assert paused == [True]
    assert graph._animations == [] and graph._interactions == []
    assert len(graph.fig.axes) == 1
    assert interaction._callback_id is None


@pytest.mark.parametrize(
    "action,message",
    [
        (lambda: kw.Graph2D().animate(lambda x, a: x, [], show=False), "at least one"),
        (lambda: kw.Graph2D().animate(lambda x, a: x, [1], samples=1, show=False), "at least 2"),
        (lambda: kw.Graph2D().animate(object(), [1], show=False), "callables, expressions"),
        (lambda: kw.Graph2D().interact(lambda x, a: x, (2, 1), show=False), "smaller"),
        (lambda: kw.Graph2D().interact(lambda x, a: x, (0, 1), initial=2, show=False), "inside"),
        (lambda: kw.Graph2D().interact(lambda x, a: x, (0, 1), step=0, show=False), "positive"),
        (lambda: kw.Graph2D().interact(lambda x, a: x, (0,), show=False), "minimum, maximum"),
        (lambda: kw.Graph2D().interact(lambda x, a: x, (0, 1), values=[0], show=False), "at least two"),
    ],
)
def test_motion_validation_errors_are_helpful(action, message):
    with pytest.raises((TypeError, ValueError), match=message):
        action()
