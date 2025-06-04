import sys
import time
from functools import wraps
from threading import Thread, Event
from typing import Callable


class Spinner:
    def __init__(self, msg='Инициализация: ', delay=0.1):
        self.msg = msg
        self.delay = delay
        self.done = Event()
        self.chars = ['|', '/', '-', '\\']
        self.thread = Thread(target=self._spin)
        self.thread.daemon = True

    def _spin(self):
        i = 0
        while not self.done.is_set():
            sys.stdout.write('\r' + self.msg + self.chars[i % len(self.chars)])
            sys.stdout.flush()
            time.sleep(self.delay)
            i += 1

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join()
        sys.stdout.write('\r')

def spinner_and_time(func: Callable, message: str):
    msg = f'{message}: '
    spinner = Spinner(msg)
    start = time.time()
    spinner.start()
    result = func()
    spinner.stop()
    elapsed = time.time() - start
    print(f'{msg} {elapsed:.2f} сек.')
    return result

def sat(message: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # msg = f'{message}: '
            # spinner = Spinner(msg)
            # start = time.time()
            # spinner.start()
            result = func(*args, **kwargs)
            # spinner.stop()
            # elapsed = time.time() - start
            # print(f'{msg}{elapsed:.2f} сек.')
            return result
        return wrapper
    return decorator
