from os import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit

AVAILABLE_PROCESSOR_CORES = cpu_count() or 1
EXECUTOR = ThreadPoolExecutor(max_workers=AVAILABLE_PROCESSOR_CORES, thread_name_prefix="solver")


def submit(fn, *args, **kwargs):
	return EXECUTOR.submit(fn, *args, **kwargs)


def map_tasks(fn, iterable, *, timeout=None, return_exceptions=False):
	futures = [EXECUTOR.submit(fn, item) for item in iterable]
	for fut in as_completed(futures, timeout=timeout):
		try:
			yield fut.result()
		except Exception as exc:
			if return_exceptions:
				yield exc
			else:
				raise


def shutdown(wait: bool = True, cancel_futures: bool = True):
	EXECUTOR.shutdown(wait=wait, cancel_futures=cancel_futures)


atexit.register(shutdown, True, True)

__all__ = [
	"AVAILABLE_PROCESSOR_CORES",
	"EXECUTOR",
	"submit",
	"map_tasks",
	"shutdown",
]
