import numpy as np
import pytest

import kiwicalc as kw


def test_public_exports_and_scalar_interface_stays_separate():
    for name in ('gradient', 'jacobian', 'hessian', 'solve_system', 'integrate_nd'):
        assert name in kw.__all__
        assert callable(getattr(kw, name))
    assert np.allclose(kw.differentiate(lambda x: x*x, at=[1, 2]), [2, 4])


@pytest.mark.parametrize('method', ['central', 'forward', 'backward'])
def test_gradient_styles_and_methods(method):
    unpacked = kw.gradient(lambda x, y: x*x + 3*y*y, at=(2, -1), method=method)
    packed = kw.gradient(lambda v: v[0]**2 + 3*v[1]**2, at=(2, -1), method=method, argument_style='vector')
    assert isinstance(unpacked, np.ndarray)
    assert np.allclose(unpacked, [4, -6], atol=1e-5)
    assert np.allclose(unpacked, packed)


def test_batched_points_have_last_axis_coordinates():
    points = np.array([[[1., 2.], [3., 4.]], [[-1., 2.], [-3., 4.]]])
    gradient = kw.gradient(lambda x, y: x*x+y*y, points)
    jacobian = kw.jacobian(lambda x, y: [x*x+y, x-y], points)
    hessian = kw.hessian(lambda x, y: x*x+y*y, points)
    assert gradient.shape == (2, 2, 2)
    assert jacobian.shape == (2, 2, 2, 2)
    assert hessian.shape == (2, 2, 2, 2)
    assert np.allclose(gradient, 2*points)
    assert np.allclose(jacobian[..., 0, 0], 2*points[..., 0])
    assert np.allclose(hessian, 2*np.eye(2), atol=1e-5)


def test_jacobian_supports_vector_outputs_and_component_functions():
    components = [lambda x, y: x*x+y, lambda x, y: x*y, lambda x, y: y*y]
    expected = [[4, 1], [3, 2], [0, 6]]
    assert np.allclose(kw.jacobian(components, at=(2, 3)), expected)
    assert np.allclose(kw.jacobian(lambda v: [v[0]**2+v[1], v[0]*v[1], v[1]**2],
                                 at=(2, 3), argument_style='vector'), expected)


def test_hessian_diagonal_mixed_partials_and_info():
    calls = []
    def f(x, y):
        calls.append((x, y))
        return x*x + 3*x*y + 4*y*y
    result = kw.hessian(f, (2, -1), return_info=True)
    assert np.allclose(result.value, [[2, 3], [3, 8]], atol=1e-5)
    assert np.array_equal(result.value, result.value.T)
    assert result.function_calls == len(calls) == 9
    assert result.converged is None and result.residual is None


def test_expression_variable_order_and_input_immutability():
    expression = kw.Poly('x^2+3*y^2')
    before = str(expression)
    assert np.allclose(kw.gradient(expression, (2, 3)), [4, 18])
    assert np.allclose(kw.gradient(expression, (3, 2), variables=('y', 'x')), [18, 4])
    assert str(expression) == before
    function = kw.Function('f(y,x)=x^2+3*y^2')
    assert np.allclose(kw.gradient(function, (3, 2)), [18, 4])
    assert np.allclose(kw.gradient(kw.Poly('x^2'), (2, 3), variables=('x', 'y')), [4, 0])
    assert np.allclose(kw.gradient(kw.Mono(3), (2, 3)), [0, 0])


def test_component_expressions_share_coordinate_order():
    assert np.allclose(kw.jacobian([kw.Poly('x^2'), kw.Poly('y^2')], (2, 3)), [[4, 0], [0, 6]])


def test_vector_callback_cannot_mutate_input_points():
    points = np.array([2., 3.])
    def f(point):
        value = np.dot(point, point)
        point[:] = 99
        return value
    assert np.allclose(kw.gradient(f, points, argument_style='vector'), [4, 6])
    assert np.array_equal(points, [2, 3])


def test_callback_errors_propagate_without_retry():
    calls = []
    def broken(x, y):
        calls.append((x, y))
        raise TypeError('inside user function')
    with pytest.raises(TypeError, match='inside user function'):
        kw.gradient(broken, (1, 2))
    assert len(calls) == 1


@pytest.mark.parametrize('at', [[], 1, [[1, 2], [3]], [np.nan, 1], [1j, 2], [True, False]])
def test_invalid_points(at):
    with pytest.raises((ValueError, TypeError)):
        kw.gradient(lambda x, y: x+y, at)


@pytest.mark.parametrize('step', [0, -1, [1], [1, 0], [np.inf, 1], True])
def test_invalid_step(step):
    with pytest.raises((ValueError, TypeError)):
        kw.hessian(lambda x, y: x*x+y*y, (1, 2), step=step)


def test_output_shape_validation():
    with pytest.raises(TypeError):
        kw.gradient(lambda x, y: [x, y], (1, 2))
    with pytest.raises(ValueError, match='vector'):
        kw.jacobian(lambda x, y: x+y, (1, 2))
    with pytest.raises(ValueError, match='remain constant'):
        kw.jacobian(lambda x, y: [x, y] if x <= 1 else [x], (1, 2))
    with pytest.raises(ValueError, match='finite'):
        kw.hessian(lambda x, y: np.inf, (1, 2))


def test_argument_style_and_variable_validation():
    with pytest.raises(ValueError):
        kw.gradient(lambda x, y: x+y, (1, 2), argument_style='auto')
    with pytest.raises(ValueError):
        kw.gradient(kw.Poly('x+y'), (1, 2), variables=('x', 'x'))
    with pytest.raises(ValueError):
        kw.gradient(kw.Poly('x+y'), (1, 2), variables=('x', 'z'))
    with pytest.raises(ValueError):
        kw.gradient(kw.Poly('x+y'), (1,))


@pytest.mark.parametrize('analytic', [False, True])
def test_solve_system_known_nonlinear_solution(analytic):
    kwargs = {'jacobian': lambda x, y: [[2*x, 2*y], [1, -1]]} if analytic else {}
    result = kw.solve_system(lambda x, y: [x*x+y*y-2, x-y], initial=(.8, 1.2), return_info=True, **kwargs)
    assert result.converged
    assert np.allclose(result.value, [1, 1])
    assert result.residual <= 1e-8
    assert result.iterations > 0 and result.function_calls > 0


def test_solver_vector_style_and_component_functions():
    expected = [2, 1]
    assert np.allclose(kw.solve_system([lambda x, y: x+y-3, lambda x, y: x-y-1], (0, 0)), expected)
    assert np.allclose(kw.solve_system(lambda v: [v.sum()-3, v[0]-v[1]-1], (0, 0),
                                       argument_style='vector', jacobian=lambda v: [[1, 1], [1, -1]]), expected)


def test_solver_damping_avoids_full_newton_overshoot():
    assert np.allclose(kw.solve_system(lambda x: [x**3-1], (.1,)), [1])


def test_solver_zero_iterations_at_solution_and_singular_failure():
    info = kw.solve_system(lambda x, y: [x-1, y-2], (1, 2), return_info=True)
    assert info.iterations == 0 and info.function_calls == 1
    info = kw.solve_system(lambda x, y: [1., 1.], (0, 0), return_info=True)
    assert not info.converged and 'singular' in info.message
    with pytest.raises(RuntimeError, match='singular'):
        kw.solve_system(lambda x, y: [1., 1.], (0, 0))
    with pytest.raises(RuntimeError, match='Maximum iterations'):
        kw.solve_system(lambda x: [x*x-2], (5,), max_iterations=1)


def test_solver_shape_and_domain_errors_are_not_hidden():
    with pytest.raises(ValueError, match='one equation'):
        kw.solve_system(lambda x, y: [x+y], (1, 2), return_info=True)
    with pytest.raises(ValueError, match='Jacobian'):
        kw.solve_system(lambda x, y: [x, y], (1, 2), jacobian=lambda x, y: [1, 2])
    with pytest.raises(ValueError, match='finite'):
        kw.solve_system(lambda x: [np.nan], (1,), return_info=True)


@pytest.mark.parametrize('method', ['midpoint', 'trapezoid'])
def test_rectangular_integration(method):
    result = kw.integrate_nd(lambda x, y: x+y, [(0, 1), (0, 2)], method=method, intervals=[4, 6], return_info=True)
    assert result.value == pytest.approx(3)
    assert result.function_calls == (24 if method == 'midpoint' else 35)
    assert result.converged is None
    assert kw.integrate_nd(lambda v: v.sum(), [(1, 0), (0, 2)], method=method,
                           argument_style='vector') == pytest.approx(-3)


def test_nd_integration_zero_width_and_guard():
    assert kw.integrate_nd(lambda x, y: pytest.fail('unexpected evaluation'), [(0, 0), (0, 1)]) == 0
    with pytest.raises(ValueError, match='max_evaluations'):
        kw.integrate_nd(lambda *x: 1, [(0, 1)]*6, intervals=20)
    with pytest.raises(ValueError, match='one count'):
        kw.integrate_nd(lambda x, y: x+y, [(0, 1), (0, 1)], intervals=[5])


def test_three_dimensional_derivatives_and_integral():
    assert np.allclose(kw.gradient(lambda x, y, z: x*y*z, (2, 3, 4)), [12, 8, 6])
    assert np.allclose(kw.hessian(lambda x, y, z: x*y*z, (2, 3, 4)),
                       [[0, 4, 3], [4, 0, 2], [3, 2, 0]], atol=1e-5)
    assert kw.integrate_nd(lambda x, y, z: x*y*z, [(0, 1)]*3, intervals=4) == pytest.approx(1/8)
