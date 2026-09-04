"""Local medians for opt-in tracing; run python -m scripts.benchmark_tracing.

Plots and HTML serialization are intentionally excluded. Times are descriptive,
not a stable CI threshold. The normal path is also guarded by regression tests.
"""
import argparse
import json
import platform
import statistics
import timeit

import kiwicalc as kw


def benchmark(number=300, repeat=7):
    cases = {
        'newton': lambda explain: kw.find_root(lambda x: x*x-2, x0=1, method='newton', explain=explain),
        'simpson_100': lambda explain: kw.integrate(lambda x: x*x, 0, 1, intervals=100, explain=explain),
    }
    results = {}
    for name, operation in cases.items():
        results[name] = {}
        for enabled in (False, True):
            times = timeit.repeat(lambda: operation(enabled), number=number, repeat=repeat)
            results[name]['traced' if enabled else 'normal'] = statistics.median(times)/number
    return {'python': platform.python_version(), 'seconds_per_call': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', type=int, default=300)
    parser.add_argument('--repeat', type=int, default=7)
    args = parser.parse_args()
    if args.number < 1 or args.repeat < 1:
        parser.error('number and repeat must be positive')
    print(json.dumps(benchmark(args.number, args.repeat), indent=2))
