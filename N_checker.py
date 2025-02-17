import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
# from VariablesPaper2 import *
# from F_inside_tank import *
# from F_tank_wall import *
# from F_inside_oriface import *
# from F_main import *

#pretty useless

def calculate_if_node_at_material_interface_int(N, d_liner, d_CFRP):
    """
    Calculate if the node at the material interface in the wall of the tank is an integer and what value it has.
    
    Args:
        N: total grid point numbers (dimensionless)
        d_liner: Thickness of the liner of the tank (m)
        d_CFRP: Thickness of the CFRP of the tank (m)
    
    Returns:
        Node that is at the material interface is an integer.
    """
    N_prime = (d_liner * (N - 1/2) + 3/2 * d_CFRP) / (d_liner + d_CFRP)
    return N_prime

def is_integer(n):
    """
    Check if the given number is an integer.
    
    Args:
        n: The number to check.
        
    Returns:
        True if n is an integer, False otherwise.
    """
    return float(n).is_integer()

def dx_finder(N_prime, d_liner):
    """
    Returns the dx needed to get a node at the interface.
    
    Args:
        n: The number to check.
        
    Returns:
        dx
    """
    return d_liner/(N_prime-3/2)

# Example usage
N = 1000
d_liner = 7*10**-3 # Thickness in meters
d_CFRP = 17*10**-3  # Thickness in meters
check = True

# Loop until we find an integer value for N_prime
while check:
    # Calculate N_prime
    N_prime = calculate_if_node_at_material_interface_int(N, d_liner, d_CFRP)

    # Check if N_prime is an integer
    if is_integer(N_prime):
        print(f"N_prime is an integer: {N_prime} (for N = {N})")
        print(f'Dx is "{d_liner/(N_prime-3/2)}')
        check = False
    else:
        N += 1
    
    
