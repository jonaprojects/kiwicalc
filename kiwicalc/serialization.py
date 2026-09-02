from __future__ import annotations

"""JSON-friendly serialization for curves, surfaces, and composed graphs."""

import json
from typing import Any

import numpy as np

from kiwicalc.core.interfaces import IExpression


def _value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _value(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _expression(value):
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, IExpression):
        return str(value)
    if callable(value):
        raise TypeError("Python callables cannot be serialized; use a string expression instead")
    raise TypeError(f"Cannot serialize expression input of type {type(value).__name__}")


def curve_to_dict(curve):
    from kiwicalc.geometry import curves as c

    common = {"samples": curve.samples}
    if isinstance(curve, c.TransformedCurve2D) or isinstance(curve, c.TransformedCurve3D):
        return {"type": type(curve).__name__, "source": curve_to_dict(curve.source), "matrix": curve.matrix.tolist()}
    if isinstance(curve, c.Ellipse):
        return {"type": "Ellipse", **common, "radius_x": curve.radius_x, "radius_y": curve.radius_y, "center": curve.center, "rotation": curve.rotation}
    if isinstance(curve, c.Arc):
        return {"type": "Arc", **common, "radius": curve.radius, "center": curve.center, "start_angle": curve.start_angle, "end_angle": curve.end_angle}
    if isinstance(curve, c.Parabola):
        return {"type": "Parabola", **common, "focal_length": curve.focal_length, "vertex": curve.vertex, "rotation": curve.rotation, "t_range": curve.t_range}
    if isinstance(curve, c.Hyperbola):
        return {"type": "Hyperbola", **common, "semi_transverse": curve.semi_transverse, "semi_conjugate": curve.semi_conjugate, "center": curve.center, "rotation": curve.rotation, "t_range": curve.t_range}
    if isinstance(curve, c.ArchimedeanSpiral):
        return {"type": "ArchimedeanSpiral", **common, "initial_radius": curve.initial_radius, "growth": curve.growth, "theta_range": curve.theta_range}
    if isinstance(curve, c.LogarithmicSpiral):
        return {"type": "LogarithmicSpiral", **common, "initial_radius": curve.initial_radius, "growth": curve.growth, "theta_range": curve.theta_range}
    if isinstance(curve, c.LissajousCurve2D):
        return {"type": "LissajousCurve2D", **common, "amplitudes": curve.amplitudes, "frequencies": curve.frequencies, "phase": curve.phase, "t_range": curve.t_range}
    if isinstance(curve, c.Cardioid):
        return {"type": "Cardioid", **common, "scale": curve.scale, "center": curve.center, "theta_range": curve.t_range}
    if isinstance(curve, c.RoseCurve):
        return {"type": "RoseCurve", **common, "radius": curve.radius, "petals": curve.petals, "theta_range": curve.theta_range}
    if isinstance(curve, c.Cycloid):
        return {"type": "Cycloid", **common, "radius": curve.radius, "turns": curve.turns, "start": curve.start}
    if isinstance(curve, (c.Epicycloid, c.Hypocycloid)):
        return {"type": type(curve).__name__, **common, "fixed_radius": curve.fixed_radius, "rolling_radius": curve.rolling_radius}
    if isinstance(curve, c.Superellipse):
        return {"type": "Superellipse", **common, "radius_x": curve.radius_x, "radius_y": curve.radius_y, "exponent": curve.exponent, "center": curve.center}
    if isinstance(curve, c.Catenary):
        return {"type": "Catenary", **common, "scale": curve.scale, "t_range": curve.t_range, "vertex": curve.vertex}
    if isinstance(curve, c.Involute):
        return {"type": "Involute", **common, "radius": curve.radius, "t_range": curve.t_range, "center": curve.center}
    if isinstance(curve, c.PolarCurve2D):
        return {"type": "PolarCurve2D", **common, "radius": _expression(curve._radius_source), "theta_range": curve.theta_range, "sampling": curve.sampling, "tolerance": curve.tolerance, "max_depth": curve.max_depth}
    if isinstance(curve, c.ParametricCurve2D):
        return {"type": "ParametricCurve2D", **common, "x": _expression(curve._x_source), "y": _expression(curve._y_source), "t_range": curve.t_range, "sampling": curve.sampling, "tolerance": curve.tolerance, "max_depth": curve.max_depth}
    if isinstance(curve, c.ImplicitCurve2D):
        return {"type": "ImplicitCurve2D", "equation": _expression(curve._equation_source), "x_range": curve.x_range, "y_range": curve.y_range, "resolution": curve.resolution, "level": curve.level}
    if isinstance(curve, c.BezierCurve2D) or isinstance(curve, c.BezierCurve3D):
        return {"type": type(curve).__name__, **common, "control_points": curve.control_points.tolist()}
    if isinstance(curve, c.CatmullRomSpline2D) or isinstance(curve, c.CatmullRomSpline3D):
        return {"type": type(curve).__name__, **common, "control_points": curve.control_points.tolist(), "closed": curve.closed}
    if isinstance(curve, c.Line3D):
        return {"type": "Line3D", **common, "point": curve.point, "direction": curve.direction, "t_range": curve.t_range}
    if isinstance(curve, c.Helix):
        return {"type": "Helix", **common, "radius": curve.radius, "pitch": curve.pitch, "turns": curve.turns, "center": curve.center}
    if isinstance(curve, c.LissajousCurve3D):
        return {"type": "LissajousCurve3D", **common, "amplitudes": curve.amplitudes, "frequencies": curve.frequencies, "phases": curve.phases, "t_range": curve.t_range}
    if isinstance(curve, c.TorusKnot):
        if isinstance(curve, c.TrefoilKnot):
            return {"type": "TrefoilKnot", **common, "major_radius": curve.major_radius, "minor_radius": curve.minor_radius}
        return {"type": "TorusKnot", **common, "p": curve.p, "q": curve.q, "major_radius": curve.major_radius, "minor_radius": curve.minor_radius}
    if isinstance(curve, c.FigureEightKnot):
        return {"type": "FigureEightKnot", **common, "scale": curve.scale}
    if isinstance(curve, c.ParametricCurve3D):
        return {"type": "ParametricCurve3D", **common, "x": _expression(curve._coordinate_sources[0]), "y": _expression(curve._coordinate_sources[1]), "z": _expression(curve._coordinate_sources[2]), "t_range": curve.t_range, "sampling": curve.sampling, "tolerance": curve.tolerance, "max_depth": curve.max_depth}
    raise TypeError(f"Unsupported curve type: {type(curve).__name__}")


def curve_from_dict(data):
    from kiwicalc.geometry import curves as c

    data = dict(data)
    type_name = data.pop("type", None)
    if type_name in ("TransformedCurve2D", "TransformedCurve3D"):
        source = curve_from_dict(data["source"])
        return source.transform(data["matrix"])
    constructors = {
        "Ellipse": c.Ellipse, "Arc": c.Arc, "Parabola": c.Parabola, "Hyperbola": c.Hyperbola,
        "ArchimedeanSpiral": c.ArchimedeanSpiral, "LogarithmicSpiral": c.LogarithmicSpiral,
        "LissajousCurve2D": c.LissajousCurve2D, "PolarCurve2D": c.PolarCurve2D,
        "Cardioid": c.Cardioid, "RoseCurve": c.RoseCurve, "Cycloid": c.Cycloid,
        "Epicycloid": c.Epicycloid, "Hypocycloid": c.Hypocycloid,
        "Superellipse": c.Superellipse, "Catenary": c.Catenary, "Involute": c.Involute,
        "ParametricCurve2D": c.ParametricCurve2D, "ImplicitCurve2D": c.ImplicitCurve2D,
        "BezierCurve2D": c.BezierCurve2D, "CatmullRomSpline2D": c.CatmullRomSpline2D,
        "Line3D": c.Line3D, "Helix": c.Helix, "LissajousCurve3D": c.LissajousCurve3D,
        "TorusKnot": c.TorusKnot, "ParametricCurve3D": c.ParametricCurve3D,
        "TrefoilKnot": c.TrefoilKnot, "FigureEightKnot": c.FigureEightKnot,
        "BezierCurve3D": c.BezierCurve3D, "CatmullRomSpline3D": c.CatmullRomSpline3D,
    }
    if type_name not in constructors:
        raise ValueError(f"Unknown curve type: {type_name!r}")
    return constructors[type_name](**data)


def surface_to_dict(surface):
    from kiwicalc.geometry import surfaces as s
    from kiwicalc.linalg.spaces import Surface

    common = {"resolution": surface.resolution}
    if isinstance(surface, Surface):
        return {"type": "Surface", **common, "coefs": [surface.a, surface.b, surface.c, surface.d]}
    if isinstance(surface, s.Sphere):
        return {"type": "Sphere", **common, "radius": surface.radius, "center": surface.center}
    if isinstance(surface, s.Ellipsoid):
        return {"type": "Ellipsoid", **common, "radii": surface.radii, "center": surface.center}
    if isinstance(surface, s.Cylinder):
        return {"type": "Cylinder", **common, "radius": surface.radius, "height": surface.height, "center": surface.center}
    if isinstance(surface, s.Cone):
        return {"type": "Cone", **common, "radius": surface.radius, "height": surface.height, "center": surface.center}
    if isinstance(surface, s.Torus):
        return {"type": "Torus", **common, "major_radius": surface.major_radius, "minor_radius": surface.minor_radius, "center": surface.center}
    if isinstance(surface, s.Paraboloid):
        return {"type": "Paraboloid", **common, "scale_x": surface.scale_x, "scale_y": surface.scale_y, "radius": surface.radius, "center": surface.center}
    if isinstance(surface, s.HyperbolicParaboloid):
        return {"type": "HyperbolicParaboloid", **common, "scale_x": surface.scale_x, "scale_y": surface.scale_y, "x_range": surface.x_range, "y_range": surface.y_range, "center": surface.center}
    if isinstance(surface, s.Hyperboloid):
        return {"type": "Hyperboloid", **common, "radii": surface.radii, "u_range": surface.u_range, "center": surface.center}
    if isinstance(surface, s.ExplicitSurface3D):
        return {"type": "ExplicitSurface3D", **common, "z": _expression(surface._z_source), "x_range": surface.x_range, "y_range": surface.y_range}
    if isinstance(surface, s.ParametricSurface3D):
        return {"type": "ParametricSurface3D", **common, "x": _expression(surface._coordinate_sources[0]), "y": _expression(surface._coordinate_sources[1]), "z": _expression(surface._coordinate_sources[2]), "u_range": surface.u_range, "v_range": surface.v_range}
    raise TypeError(f"Unsupported surface type: {type(surface).__name__}")


def surface_from_dict(data):
    from kiwicalc.geometry import surfaces as s
    from kiwicalc.linalg.spaces import Surface

    data = dict(data)
    type_name = data.pop("type", None)
    constructors = {
        "Sphere": s.Sphere, "Ellipsoid": s.Ellipsoid, "Cylinder": s.Cylinder,
        "Cone": s.Cone, "Torus": s.Torus, "ExplicitSurface3D": s.ExplicitSurface3D,
        "ParametricSurface3D": s.ParametricSurface3D,
        "Paraboloid": s.Paraboloid, "HyperbolicParaboloid": s.HyperbolicParaboloid,
        "Hyperboloid": s.Hyperboloid,
        "Surface": Surface,
    }
    if type_name not in constructors:
        raise ValueError(f"Unknown surface type: {type_name!r}")
    return constructors[type_name](**data)


def object_to_dict(obj):
    from kiwicalc.functions.function import Function
    from kiwicalc.geometry.curves import Curve2D, Curve3D
    from kiwicalc.geometry.points import Circle, Point2D, Point3D
    from kiwicalc.geometry.surfaces import Surface3D
    from kiwicalc.linalg.spaces import Surface

    if isinstance(obj, Surface):
        return {"kind": "plane", "coefficients": [obj.a, obj.b, obj.c, obj.d], "resolution": obj.resolution}
    if isinstance(obj, (Curve2D, Curve3D)):
        return {"kind": "curve", "data": curve_to_dict(obj)}
    if isinstance(obj, Surface3D):
        return {"kind": "surface", "data": surface_to_dict(obj)}
    if isinstance(obj, Function):
        return {"kind": "function", "expression": obj.function_string}
    if isinstance(obj, IExpression):
        return {"kind": "expression", "expression": str(obj)}
    if isinstance(obj, Circle):
        radius = obj.radius.try_evaluate()
        center = (obj.center_x.try_evaluate(), obj.center_y.try_evaluate())
        if radius is None or None in center:
            raise TypeError("Only numeric circles can be serialized")
        return {"kind": "circle", "radius": radius, "center": center}
    if isinstance(obj, (Point2D, Point3D)):
        return {"kind": "point", "coordinates": list(obj.coordinates)}
    if isinstance(obj, str):
        return {"kind": "string", "value": obj}
    if isinstance(obj, (int, float)):
        return {"kind": "number", "value": obj}
    if callable(obj):
        raise TypeError("Python callables cannot be serialized; use a string or KiwiCalc Function")
    raise TypeError(f"Unsupported graph item: {type(obj).__name__}")


def object_from_dict(data):
    from kiwicalc.expressions.factory import create
    from kiwicalc.functions.function import Function
    from kiwicalc.geometry.points import Circle, Point2D, Point3D
    from kiwicalc.linalg.spaces import Surface

    kind = data.get("kind")
    if kind == "curve": return curve_from_dict(data["data"])
    if kind == "surface": return surface_from_dict(data["data"])
    if kind == "plane": return Surface(data["coefficients"], resolution=data.get("resolution", 80))
    if kind == "function": return Function(data["expression"])
    if kind == "expression": return create(data["expression"])
    if kind == "circle": return Circle(data["radius"], data["center"])
    if kind == "point":
        return Point2D(*data["coordinates"]) if len(data["coordinates"]) == 2 else Point3D(*data["coordinates"])
    if kind in ("string", "number"): return data["value"]
    raise ValueError(f"Unknown serialized object kind: {kind!r}")


def graph_to_dict(graph):
    from kiwicalc.plotting.plots import Graph2D, Graph3D

    if not isinstance(graph, (Graph2D, Graph3D)):
        raise TypeError("Only Graph2D and Graph3D can be serialized")
    if graph._secondary_axis_specs:
        raise TypeError("Secondary axes use Python callables and cannot be serialized")
    entries = []
    for item, options in graph._entries():
        entries.append({"object": object_to_dict(item), "label": options["label"], "visible": options["visible"], "style": _value(options["style"])})
    decorations = []
    for decoration in graph._decorations:
        encoded = dict(decoration)
        object_fields = {"first", "second"} if encoded["kind"] == "fill" else {"source", "first", "second", "other", "u", "v"}
        for field in object_fields & encoded.keys():
            encoded[field] = object_to_dict(encoded[field])
        decorations.append(_value(encoded))
    view = {}
    if graph._has_plotted:
        view = {
            "title": graph.ax.get_title(), "xlim": graph.ax.get_xlim(), "ylim": graph.ax.get_ylim(),
            "xlabel": graph.ax.get_xlabel(), "ylabel": graph.ax.get_ylabel(),
            "legend": graph.ax.get_legend() is not None,
            "grid": any(line.get_visible() for line in (*graph.ax.get_xgridlines(), *graph.ax.get_ygridlines())),
            "equal_aspect": graph.ax.get_aspect() == 1.0,
            "xscale": graph.ax.get_xscale(), "yscale": graph.ax.get_yscale(),
            "axis_options": _value(graph._axis_options),
        }
        if graph._theme is not None:
            view["theme"] = graph._theme.to_dict()
    if isinstance(graph, Graph3D) and graph._has_plotted:
        view["zlim"] = graph.ax.get_zlim()
        view.update(zlabel=graph.ax.get_zlabel(), zscale=graph.ax.get_zscale())
    return _value({"type": type(graph).__name__, "items": entries, "decorations": decorations, "view": view})


def graph_from_dict(data):
    from kiwicalc.plotting.plots import Graph2D, Graph3D

    graph_type = data.get("type")
    if graph_type not in ("Graph2D", "Graph3D"):
        raise ValueError(f"Unknown graph type: {graph_type!r}")
    graph = Graph2D() if graph_type == "Graph2D" else Graph3D()
    for entry in data.get("items", []):
        graph.add(object_from_dict(entry["object"]), label=entry.get("label"), visible=entry.get("visible", True), **entry.get("style", {}))
    for decoration in data.get("decorations", []):
        kind = decoration["kind"]
        if kind == "annotation": graph.annotate(decoration["text"], decoration["at"], decoration["offset"], **decoration.get("style", {}))
        elif kind == "vertical": graph.vertical_line(decoration["value"], decoration.get("label"), **decoration.get("style", {}))
        elif kind == "horizontal": graph.horizontal_line(decoration["value"], decoration.get("label"), **decoration.get("style", {}))
        elif kind == "fill": graph.fill_between(object_from_dict(decoration["first"]), object_from_dict(decoration["second"]), decoration.get("values"), decoration.get("label"), **decoration.get("style", {}))
        else:
            from kiwicalc.plotting.explanations import EXPLANATION_KINDS
            from kiwicalc.plotting.fields import FIELD_KINDS
            if kind not in EXPLANATION_KINDS | FIELD_KINDS:
                raise ValueError(f"Unknown graph decoration kind: {kind!r}")
            restored = dict(decoration)
            for field in {"source", "first", "second", "other", "u", "v"} & restored.keys():
                restored[field] = object_from_dict(restored[field])
            graph._decorations.append(restored)
    graph._restored_view = dict(data.get("view", {}))
    return graph


def dumps(obj, **kwargs):
    if hasattr(obj, "to_dict"):
        return json.dumps(obj.to_dict(), **kwargs)
    raise TypeError(f"{type(obj).__name__} does not support KiwiCalc serialization")
