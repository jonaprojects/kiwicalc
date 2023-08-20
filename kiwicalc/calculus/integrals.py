def reinman(f: Callable, a, b, N: int):
    if N < 2:
        raise ValueError("The method requires N >= 2")
    return sum((b - a) / (N - 1) * f(value) for value in np.linspace(a, b, N))


def trapz(f: Callable, a, b, N: int):
    if N == 0:
        raise ValueError("Trapz(): N cannot be 0")
    dx = (b - a) / N
    return 0.5 * dx * sum((f(a + i * dx) + f(a + (i - 1) * dx)) for i in range(1, int(N) + 1))


def simpson(f: Callable, a, b, N: int):
    if N <= 2:
        raise ValueError("The method requires N >= 2")
    dx = (b - a) / (N - 1)
    if N % 2 != 0:
        N += 1
    return (dx / 3) * sum(
        ((f(a + (2 * i - 2) * dx) + 4 * f(a + (2 * i - 1) * dx) + f(a + 2 * i * dx)) for i in range(1, int(N / 2))))
