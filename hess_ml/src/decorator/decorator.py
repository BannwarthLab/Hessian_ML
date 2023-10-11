import time as time 

def checkTiming(func):
    def wrapper(self,*args, **kwargs):
        cur_time = time.time()
        result = func(self,*args, **kwargs)
        print(f'{func.__name__} performed in {time.time()- cur_time: 4.5f} s ')
        return result
    return wrapper
