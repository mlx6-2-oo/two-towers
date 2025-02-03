import pytest
from math_utils import add, multiply

def test_add():
    """Test the add function with various inputs."""
    # Test positive numbers
    assert add(2, 3) == 5
    
    # Test negative numbers
    assert add(-1, -1) == -2
    
    # Test zero
    assert add(0, 5) == 5
    
    # Test floating point numbers
    assert add(1.5, 2.5) == 4.0

def test_multiply():
    """Test the multiply function with various inputs."""
    # Test positive numbers
    assert multiply(2, 3) == 6
    
    # Test negative numbers
    assert multiply(-2, 3) == -6
    
    # Test zero
    assert multiply(0, 5) == 0
    
    # Test floating point numbers
    assert multiply(1.5, 2) == 3.0 