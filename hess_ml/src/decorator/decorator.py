import time


def checkTiming(enabled=True):
    def Timing(func):
        def wrapper(self, *args, **kwargs):
            if enabled:
                cur_time = time.time()
                result = func(self, *args, **kwargs)
                print(f"{func.__name__} performed in {time.time()- cur_time: 4.5f} s ")
                return result

            return func(self, *args, **kwargs)
        return wrapper

    return Timing

def check_calc(func):
    def wrapper(self, *args, **kwargs):
        if self.do_calc:
            return func(self, *args, **kwargs)
        else:
            return None 
    return wrapper


def initProcess(func):
    def wrapper(self, *args, **kwargs):
        print("\n")
        print(f"Starting {func.__name__}")
        print("*" * 14)
        print("\n")
        result = func(self, *args, **kwargs)
        print("\n")
        print("*" * 14)
        print("\n")
        return result

    return wrapper
