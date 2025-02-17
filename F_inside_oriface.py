import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from F_inside_tank import *
from F_tank_wall import *
from F_main import *

# Functions Section 2.2 

# Function to calculate density at the orifice using isentropic expansion
def calculate_density_orifice_rho2(rho1, b, gamma):
    """
    Calculate density at the orifice using the isentropic expansion relation.
    Args:
        rho1: Density of the gas inside the tank (kg/m3)
        b: Co-volume constant (m3/kg)
        gamma: Specific heat ratio (-)
    Returns:
        Density at the orifice (kg/m3)
    """
    # Implementing the transcendental equation for density at the orifice
    #Newton considered so far but others may work more precisely, look at later
    def func(rho2):
        return ((rho1 / (1 - b * rho1))) - ((rho2 / (1 - b * rho2))) * (1 + ((gamma - 1) / (2 * (1 - b * rho2) ** 2))) ** (1 / (gamma - 1))
    
    return newton(func, rho1 * 0.5)  # Initial guess is half of rho1

#Function to calculate the temperature T2 at the orafice using energy conservation
def calculate_temperature_energy_conservation_T2(rho2, T1, b, gamma):
    """
    Calculate temperature of the gas at the orafice using conservation of energy between inside of
    tank and orifice. Assumes Isentropic flow.
    Args:
        rho2: Density of the gas at the orafice (kg/m3)
        T1: Temperature inside the tank (K)
        b: Co-volume constant (m3/kg)
        gamma: Specific heat ratio (-)
    Returns:
        Temperature at the orafice (K)
    """
    return T1*(2*(1-b*rho2)**2)/(2*(1-b*rho2)**2+gamma-1)

# Function to calculate pressure at the oriface using Abel-Noble equation
def calculate_pressure_abel_noble_P2(T2, rho2, b, R_g):
    """
    Calculate temperature of the gas inside the tank using Abel-Noble equation.
    Args:
        T2 : Temperature at the orafice (K)
        rho2: Density of the gas at the orafice (kg/m3)
        b: Co-volume constant (m3/kg)
        R_g: Gas constant (m2 s2 K^-1)
    Returns:
        Temperature inside the tank (K)
    """
    return rho2 * R_g * T2 / (1-b*rho2)

# Function to calculate velocity at the orifice and notional nozzle
def calculate_velocity(T, gamma, R_g, rho, b):
    """
    Calculate velocity at the orifice or notional nozzle using sound velocity equation.
    Args:
        T: Temperature at the orifice or nozzle (K)
        gamma: Specific heat ratio (-)
        R_g: Gas constant (m2 s2 K^-1)
        rho: Density at the orifice or nozzle (kg/m3)
    Returns:
        Velocity (m/s)
    """
    return np.sqrt(gamma * R_g * T) / (1 - b * rho)

#Function to calculate the temperature T3 at the notional nozzle using energy conservation
def calculate_temperature_energy_conservation_T3(P2, rho2, T2, b, R_g, gamma):
    """
    Calculate temperature of the gas at the notional nozzle using energy conservation.
    Assume: velocity of gas at notional nozzle equal to local speed of sound.
    Args:
        P2: Pressure at the orafice (Pa)
        rho2: Density of the gas at the orafice (kg/m3)
        T2 : Temperature at the orafice (K)
        b: Co-volume constant (m3/kg)
        R_g: Gas constant (m2 s2 K^-1)
        gamma: Specific heat ratio (-)
    Returns:
        Temperature at the notional nozzle (K)
    """
    return 2*T2/(gamma+1)+(gamma-1)/(gamma+1)*P2/(rho2*(1-b*rho2)*R_g)

# Function to calculate density at the notional nozzle using Abel-Noble equation
def calculate_density_abel_noble_rho3(T3, P_amb, b, R_g):
    """
    Calculate temperature of the gas inside the tank using Abel-Noble equation.
    Assume: ambient pressure at notional nozzle.
    Args:
        T3 : Temperature at the notional nozzle (K)
        P_amb : Ambient pressure (Pa)
        b: Co-volume constant (m3/kg)
        R_g: Gas constant (m2 s2 K^-1)
    Returns:
        Density at the notional nozzle (kg/m3)
    """
    return P_amb / (P_amb*b + R_g*T3)

# Function to calculate the diameter of the notional nozzle (D3)
def calculate_notional_nozzle_diameter(D2_orifice, CD, rho2, u2, rho3, u3):
    """
    Calculate the diameter of the notional nozzle.
    Args:
        D2_orifice: Diameter of the orifice (m)
        CD: Discharge coefficient (-)
        rho2: Density at the orifice (kg/m3)
        u2: Velocity at the orifice (m/s)
        rho3: Density at the notional nozzle (kg/m3)
        u3: Velocity at the notional nozzle (m/s)
    Returns:
        Diameter of the notional nozzle (m)
    """
    return D2_orifice * np.sqrt(CD * (rho2 * u2) / (rho3 * u3))

# Function to calculate mass flow rate (mass_flow_rate)
def calculate_mass_flow_rate(rho3, u3, D3):
    """
    Calculate the mass flow rate at the notional nozzle.
    Args:
        rho3: Density at the notional nozzle (kg/m3)
        u3: Velocity at the notional nozzle (m/s)
        D3: Diameter of the notional nozzle (m)
    Returns:
        Mass flow rate (kg/s)
    """
    return rho3 * u3 * np.pi * (D3 ** 2) / 4