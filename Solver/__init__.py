from os import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Expose a thread pool sized to the number of available CPU cores, similar to:
# private final int AvailableProcessorCors = ManagementFactory.getOperatingSystemMXBean().getAvailableProcessors();
# private final ExecutorService Executor = Executors.newFixedThreadPool(this.AvailableProcessorCors);


AVAILABLE_PROCESSOR_CORES: int = cpu_count() or 1
EXECUTOR: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=AVAILABLE_PROCESSOR_CORES)


def submit(fn, *args, **kwargs):
    """Submit a callable to the shared executor."""
    return EXECUTOR.submit(fn, *args, **kwargs)


def map_tasks(fn, iterable, *, timeout=None, return_exceptions=False):
    """
    Execute fn over iterable using the shared executor.
    Yields results in completion order.
    """
    futures = [EXECUTOR.submit(fn, item) for item in iterable]
    for fut in as_completed(futures, timeout=timeout):
        try:
            yield fut.result()
        except Exception as exc:
            if return_exceptions:
                yield exc
            else:
                raise


def shutdown(wait: bool = True, cancel_futures: bool = False):
    """Gracefully shut down the shared executor."""
    EXECUTOR.shutdown(wait=wait, cancel_futures=cancel_futures)


__all__ = [
    "AVAILABLE_PROCESSOR_CORES",
    "EXECUTOR",
    "submit",
    "map_tasks",
    "shutdown",
]