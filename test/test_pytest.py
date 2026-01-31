import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from calculator import fun1, fun2, fun3, fun4

def test_fun1():
    """Test square root of sum"""
    assert fun1(9, 16) == 5.0
    assert fun1(0, 25) == 5.0

def test_fun2():
    """Test power function"""
    assert fun2(2, 3) == 8
    assert fun2(5, 2) == 25

def test_fun3():
    """Test percentage function"""
    assert fun3(25, 50) == 50.0
    assert fun3(10, 100) == 10.0

def test_fun4():
    """Test hypotenuse function"""
    assert fun4(3, 4) == 5.0
    assert fun4(6, 8) == 10.0