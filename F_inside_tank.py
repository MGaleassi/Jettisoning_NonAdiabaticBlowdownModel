import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from F_inside_oriface import *
from F_tank_wall import *
from F_main import *

# Functions Section 2.1 

# Function to calculate heat transfer from surroundings
def calculate_heat_transfer(k_int, A_int, T_w_int, T1):
    """
    Calculate heat transfer from the tank wall to the inside gas.
    Args:
        k_int: Heat transfer coefficient between internal gas and internal wall (W/m2/K)
        A_int: Internal surface area of the tank (m2)
        T_w_int: Temperature of the inside surface of the tank wall (K)
        T1: Temperature inside the tank (K)
    Returns:
        Heat transfer rate (J/s)
    """
    return k_int * A_int * (T_w_int - T1)

# Function to calculate internal energy in the tank
def calculate_internal_energy(P1, V, m, b, gamma):#use only once for initial value
    """
    Calculates the internal energy of the gas inside the tank.
    Args:
        P1: Pressure inside the tank (Pa)
        V: Volume of the tank (m3)
        m: Mass of gas inside the tank (kg)
        b: Co-volume constant (m3/kg)
        gamma: Specific heat ratio (-)
    Returns:
        Internal energy in the tank (J)
    """
    return (P1 * (V - m * b)) / (gamma - 1)

# Function to calculate P1 from internal energy in the tank
def calculate_internal_pressure_P1_from_internal_energy(U, V, m, b, gamma):
    """
    Calculates the internal energy of the gas inside the tank.
    Args:
        U: Internal energy in the tank (J)
        V: Volume of the tank (m3)
        m: Mass of gas inside the tank (kg)
        b: Co-volume constant (m3/kg)
        gamma: Specific heat ratio (-)
    Returns:
        Pressure inside the tank (Pa)
    """
    return (gamma - 1)*U/(V-m*b)

# Function to calculate the initial density in the tank using Abel-Noble equation
def calculate_density_abel_noble_rho1_initial(T1, P1, b, R_g):
    """
    Calculate temperature of the gas inside the tank using Abel-Noble equation.
    Assume: ambient pressure at notional nozzle.
    Args:
        T1 : Temperature in the tank (K)
        P1 : Pressure in the tank (Pa)
        b: Co-volume constant (m3/kg)
        R_g: Gas constant (m2 s2 K^-1)
    Returns:
        Initial denisty in the tank at time 0 (kg/m3)
    """
    return P1 / (P1*b + R_g*T1)

# Function to calculate temperature inside the tank using Abel-Noble equation
def calculate_temperature_abel_noble_T1(P1, rho1, b, R_g):
    """
    Calculate temperature of the gas inside the tank using Abel-Noble equation.
    Args:
        P1: Pressure inside the tank (Pa)
        rho1: Density of the gas inside the tank (kg/m3)
        b: Co-volume constant (m3/kg)
        R_g: Gas constant (m2 s2 K^-1)
    Returns:
        Temperature inside the tank (K)
    """
    return P1 * (1 - b * rho1) / (rho1 * R_g)