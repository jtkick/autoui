#!/usr/bin/env python3

def test():
    print("yolo")

class Test:

    def __init__(self, func):
        self._func = func

    def __enter__(self):
        print('enter')
        return self

    def __exit__(self, exc_type, exc_value, tb):
        print('exit')

    def run(self):
        self._func()
    
with Test(test) as t:
    t.run()