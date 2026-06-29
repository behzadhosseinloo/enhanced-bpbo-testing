"""
Triangle Type Classification Program
Benchmark for Test Data Generation
"""

def triangle_type(a, b, c):
    """
    Triangle classification program.
    Returns the path ID (1-4) and the classification result.
    """
    # Validate inputs
    if a <= 0 or b <= 0 or c <= 0:
        return 1, "Invalid (negative/zero side)"
    
    # Check triangle inequality
    if a + b <= c or a + c <= b or b + c <= a:
        return 1, "Invalid (inequality)"
    
    # Classify triangle
    if a == b and b == c:
        return 4, "Equilateral"
    elif a == b or b == c or a == c:
        return 3, "Isosceles"
    else:
        return 2, "Scalene"
