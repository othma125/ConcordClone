#!/usr/bin/env python3
"""Simple console to run solvers and plot tours.

Commands:
 - solve <file> [<method>] [<max_time>]
 - plot [<file> [<method>] [<max_time>]]   # if args omitted, plots last solved tour
 - exit | quit

This CLI uses TSPSolver and the Methods.registry.
"""
import shlex
from pathlib import Path
from typing import Optional, List

from TSPSolver.TSPTour import TSPTour

# try to enable an input function that supports command history
# Prefer system readline (POSIX), then prompt_toolkit (cross-platform). Fall back to builtin input.
readline = None
input_func = input

LAST_SOLVED = {}

REPO_ROOT = Path(__file__).resolve().parent
TSPLIB_DIR = REPO_ROOT / 'DefaultInstances' / 'TSPLIB'


def solve_command(file_name: str, method: Optional[str] = None, max_time: Optional[float] = None) -> None:
    """Solve a TSP file. method and max_time are optional and will use solver defaults when None.

    Returns the tour object or None on failure.
    """
    # import lazily to avoid circular imports at module import time
    global kwargs
    from TSPSolver.TSPSolver import TSPSolver
    from TSPData.TSPInstance import TSPInstance
    from TSPSolver.Methods import registry

    if method is not None and method not in registry:
        print(f"Unknown method: {method}. Available: {', '.join(sorted(registry.keys()))}")
        return None

    # resolve TSP file: accept name with or without .tsp and case-insensitive matches
    tsp_path = resolve_tsp_path(file_name)
    if tsp_path is None:
        print(f"TSP file not found in {TSPLIB_DIR}: {file_name}")
        return None

    print(f"Solving {file_name} with method={method or '(default)'} max_time={max_time if max_time is not None else '(default)'}")
    data = TSPInstance(str(tsp_path))
    solver = TSPSolver(data)
    # try:
    kwargs = {}
    if method is not None:
        kwargs['method'] = method
    if max_time is not None:
        kwargs['max_time'] = max_time
    tour = solver.Solve(**kwargs)
    # except Exception as e:
    #     # Helpful fallback: if a required library is missing (networkx/matplotlib),
    #     # try a different solver (pyvrp_hgs) if available.
    #     err = str(e)
    #     print(f"Solver error: {e}")
    #     registry = None
    #     try:
    #         from TSPSolver.Methods import registry as _registry
    #         registry = _registry
    #     except Exception:
    #         registry = None

    #     if (isinstance(e, ImportError) or 'networkx' in err.lower() or 'matplotlib' in err.lower()) and registry is not None:
    #         if 'pyvrp_hgs' in registry and (method is None or method != 'pyvrp_hgs'):
    #             print("Attempting fallback solver 'pyvrp_hgs' because of missing optional dependency...")
    #             try:
    #                 kwargs['method'] = 'pyvrp_hgs'
    #                 tour = solver.Solve(**kwargs)
    #                 print("Fallback to 'pyvrp_hgs' succeeded.")
    #                 LAST_SOLVED['tour'] = tour
    #                 LAST_SOLVED['solver'] = solver
    #                 print(tour)
    #                 return tour
    #             except Exception as e2:
    #                 print(f"Fallback solver also failed: {e2}")
    #     return
    
    LAST_SOLVED['tour'] = tour
    LAST_SOLVED['solver'] = solver
    print(tour)
    

def plot_command(args: List[str]):
    """Plot last tour or solve+plot with optional args.

    Usage:
      plot                -> plot last solved tour
      plot file           -> solve file with defaults then plot
      plot file method    -> solve file with method then plot
      plot file method max_time -> solve file with method and max_time then plot
    """
    # no args -> plot last solved
    if not args:
        if 'tour' not in LAST_SOLVED:
            print('No previously solved tour to plot.')
            return
        solver = LAST_SOLVED.get('solver')
        solver.Visualisation(LAST_SOLVED['tour'])
        return

    # parse optional args
    if len(args) > 3:
        print('Usage: plot [<file> [<method> [<max_time>]]])')
        return

    file_name = args[0]
    method = args[1] if len(args) >= 2 else None
    max_time = float(args[2]) if len(args) == 3 else None

    solve_command(file_name, method, max_time)
    solver = LAST_SOLVED.get('solver')
    solver.Visualisation(LAST_SOLVED['tour'])


def resolve_tsp_path(file_name: str) -> Optional[Path]:
    """Resolve a file name under TSPLIB_DIR.

    Accepts names with or without extension, and does a case-insensitive stem match.
    Returns a Path if found, otherwise None.
    """
    # direct path under TSPLIB_DIR
    p = TSPLIB_DIR / file_name
    if p.is_file():
        return p
    # try adding .tsp when missing
    p2 = TSPLIB_DIR / (file_name + '.tsp')
    if p2.is_file():
        return p2
    # try case-insensitive stem match
    base = Path(file_name).stem.lower()
    try:
        for child in TSPLIB_DIR.iterdir():
            if child.is_file() and child.stem.lower() == base:
                return child
    except FileNotFoundError:
        return None
    return None


def repl():
    print('Console started. Type help for commands.')
    while True:
        try:
            # use the chosen input function; prompt text works for both input() and prompt_toolkit
            line = input_func('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nExiting.')
            return
        if not line:
            continue
        parts = shlex.split(line)
        cmd = parts[0].lower()
        args = parts[1:]
        if cmd in ('exit', 'quit'):
            print('Bye')
            return
        if cmd == 'help':
            print(__doc__)
            continue
        if cmd == 'solve':
            if len(args) < 1 or len(args) > 3:
                print('Usage: solve <file> [<method>] [<max_time>]')
                continue
            file_name = args[0]
            method = args[1] if len(args) >= 2 else None
            max_time = float(args[2]) if len(args) == 3 else None
            solve_command(file_name, method, max_time)
            continue
        if cmd == 'plot':
            plot_command(args)
            continue
        print('Unknown command. Type help.')


if __name__ == '__main__':
    repl()
