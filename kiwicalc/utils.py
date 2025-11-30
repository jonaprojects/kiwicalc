"""
Utility functions for kiwicalc
"""

def round_decimal(number: float):
    """
    Round a decimal number to a reasonable number of decimal places.
    
    Args:
        number: The number to round
        
    Returns:
        The rounded number
    """
    if number == 0:
        return 0
    if abs(number) >= 1:
        return round(number, 6)
    else:
        return round(number, 10)
