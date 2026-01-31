import math

def fun1(x, y):
    """Returns the square root of the sum of x and y"""
    return math.sqrt(x + y)

def fun2(x, y):
    """Returns x raised to the power of y"""
    return x ** y

def fun3(x, y):
    """Returns the percentage: what percent x is of y"""
    if y == 0:
        return "Error: Division by zero"
    return (x / y) * 100

def fun4(x, y):
    """Returns the hypotenuse of a right triangle with sides x and y"""
    return math.sqrt(x**2 + y**2)