import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return out

    return wrapper
