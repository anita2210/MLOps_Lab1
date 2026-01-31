import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from calculator import fun1, fun2, fun3, fun4

class TestCalculator(unittest.TestCase):
    
    def test_fun1(self):
        """Test square root of sum"""
        self.assertEqual(fun1(9, 16), 5.0)
        self.assertEqual(fun1(0, 25), 5.0)
    
    def test_fun2(self):
        """Test power function"""
        self.assertEqual(fun2(2, 3), 8)
        self.assertEqual(fun2(5, 2), 25)
    
    def test_fun3(self):
        """Test percentage function"""
        self.assertEqual(fun3(25, 50), 50.0)
        self.assertEqual(fun3(10, 100), 10.0)
    
    def test_fun4(self):
        """Test hypotenuse function"""
        self.assertEqual(fun4(3, 4), 5.0)
        self.assertEqual(fun4(6, 8), 10.0)

if __name__ == '__main__':
    unittest.main()