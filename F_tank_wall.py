import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from F_inside_tank import *
from F_inside_oriface import *
from F_main import *

# Functions Section 2.3 

# Function to calculate Rayleigh number inside the tank (Ra_int)
def calculate_rayleigh_number_internal(g, beta, T1, T_w_int, cp_gas, rho1, D_int, mu_g, lambda_g):
    """
    Calculate Rayleigh number for internal gas convection inside the tank.
    Args:
        g: Acceleration due to gravity (m/s^2)
        beta: Thermal expansion coefficient of the gas (1/K)
        T1: Temperature inside the tank (K)
        T_w_int: Temperature of the inside surface of the tank wall (K)
        cp_gas: Specific heat capacity of the gas (J/kg/K)
        rho1: Density of the gas inside the tank (kg/m3)
        D_int: Internal diameter of the tank (m)
        mu_g: Dynamic viscosity of the gas (Pa.s)
        lambda_g: Thermal conductivity of the gas (W/m/K)
    Returns:
        Rayleigh number (dimensionless)
    """
    return (g * beta * abs(T1 - T_w_int) * cp_gas * (rho1 ** 2) * (D_int ** 3)) / (mu_g * lambda_g)


# # Function to calculate external Rayleigh number (Ra_din)
# def calculate_rayleigh_number_external(g, beta, T_w_ext, T_amb, L_ext, nu_air, alpha_air):
#     """
#     Calculate Rayleigh number for external natural convection.
#     Args:
#         g: Acceleration due to gravity (m/s^2)
#         beta: Thermal expansion coefficient of the air (1/K)
#         T_w_ext: Temperature of the external surface of the tank (K)
#         T_amb: Ambient temperature (K)
#         L_ext: Characteristic length (m)
#         nu_air: Kinematic viscosity of the air (m2/s)
#         alpha_air: Thermal diffusivity of the air (m2/s)
#     Returns:
#         Rayleigh number (dimensionless)
#     """
#     return (g * beta * abs(T_w_ext - T_amb) * L_ext ** 3) / (nu_air * alpha_air)

# Function to calculate Nusselt number inside the tank (Nu_Din)
def calculate_nusselt_number_internal(Ra_int):
    """
    Calculate the Nusselt number for internal convection inside the tank.
    Args:
        Ra_int: Rayleigh number of the inside gas (-)
    Returns:
        Nusselt number (dimensionless)
    """
    return 0.104 * (Ra_int ** 0.352)

# Function to calculate Nusselt number for external convection (Nu_Dext)
def calculate_nusselt_number_external(Re_Dext, Pr_air):
    """
    Calculate Nusselt number for external convection.
    Args:
        Re_Dext: Reynolds unmber of air (-)
        Pr_air: Prandtl number for air (-)
    Returns:
        Nusselt number (dimensionless)
    """
    return (0.4*Re_Dext**0.5+0.06*Re_Dext**2/3)*Pr_air**0.4

# Function to calculate Reynolds number of air (Re_Dext)
def calculate_reynolds_number_external(rho_air, V_air, D_ext, mu_air):
    """
    Calculate Reynolds number for external forced convection.
    Args:
        rho_air: Air density (kg/m3)
        V_air: Velocity of air (m/s)
        D_ext: Characteristic length (external diameter of tank) (m)
        mu_air: Dynamic viscosity of the air (Pa.s)
    Returns:
        Reynolds number (dimensionless)
    """
    return (rho_air*V_air * D_ext) / mu_air

# Function to calculate Prandtl number for air (Pr_air)
def calculate_prandtl_number(mu_air, cp_air, lambda_air):
    """
    Calculate Prandtl number for air.
    Args:
        cp_air: Specific heat of air (J/kg/K)
        mu_air: Dynamic viscosity of air (Pa.s)
        lambda_air: Thermal conductivity of air (W/m/K)
    Returns:
        Prandtl number (dimensionless)
    """
    return (mu_air*cp_air)/lambda_air

# Function to calculate internal surface heat transfer coefficient (k_int)
def calculate_internal_surface_heat_transfer_coefficient(Nu_Din, lambda_g, D_int):
    """
    Calculate Prandtl number for air.
    Args:
        Nu_Din: Nusselt number for internal convection (dimensionless)
        lambda_g: Thermal conductivity of the gas (W/m·K)
        D_int: Internal diameter of the tank (m)
    Returns:
        Heat transfer coefficient of the internal surface of the wall (W/m²·K)
    """
    return (lambda_g*Nu_Din)/D_int

# Function to calculate external surface heat transfer coefficient (k_ext)
def calculate_external_surface_heat_transfer_coefficient(Nu_Dext, lambda_air, D_ext):
    """
    Calculate Prandtl number for air.
    Args:
        Nu_Dext: Nusselt number for external convection (dimensionless)
        lambda_air: Thermal conductivity of the ambient air (W/m·K)
        D_ext: External diameter of the tank (m)
    Returns:
        Heat transfer coefficient of the external surface of the wall (W/m²·K)
    """
    return (lambda_air*Nu_Dext)/D_ext

def weighted_average_density(L_liner, L_CFRP, rho_liner, rho_CFRP):
    """
    Calculate the weighted average density of the wall material.
    Args:
        L_liner: Thickness of the liner (m)
        L_CFRP: Thickness of the CFRP (m)
        rho_liner: Density of the liner (kg/m3)
        rho_CFRP: Density of the CFRP (kg/m3)
    Returns:
        Weighted average density of the wall material (kg/m3)
    """
    total_thickness = L_liner + L_CFRP
    rho_w_n = (L_liner * rho_liner + L_CFRP * rho_CFRP) / total_thickness
    return rho_w_n

def weighted_average_cp(L_liner, L_CFRP, cp_liner, cp_CFRP):
    """
    Calculate the weighted average specific heat capacity of the wall material.
    Args:
        L_liner: Thickness of the liner (m)
        L_CFRP: Thickness of the CFRP (m)
        cp_liner: Specific heat capacity of the liner (J/kg/K)
        cp_CFRP: Specific heat capacity of the CFRP (J/kg/K)
    Returns:
        Weighted average specific heat capacity of the wall material (J/kg/K)
    """
    total_thickness = L_liner + L_CFRP
    cp_wall_n = (L_liner * cp_liner + L_CFRP * cp_CFRP) / total_thickness
    return cp_wall_n

# Finite difference solver for 1D unsteady heat conduction within tank wall
def heat_conduction_solver(rho_w_n, cp_wall_n, 
                           rho_liner, rho_CFRP, cp_liner, cp_CFRP, 
                           #lambda_ext_minus, lambda_ext_plus, lambda_int_minus, lambda_int_plus, 
                           #delta_x_ext_minus, delta_x_ext_plus, delta_x_int_minus, delta_x_int_plus, 
                           Tw_prev, dx, dt, k_int, k_ext, T1, T_amb,
                           N_prime, lambda_liner, lambda_CFRP):
    """
    Solves the unsteady heat conduction equation within the tank wall using the finite difference method.
    Args:
        rho_w_n: Density of the wall material (kg/m3)
        cp_wall_n: Specific heat capacity of the wall material (J/kg/K)
        lambda_ext_minus: Thermal conductivity from grid point n to the external surface (W/m/K)
        lambda_ext_plus: Thermal conductivity from the external surface to grid point n+1 (W/m/K)
        lambda_int_minus: Thermal conductivity from grid point n to the internal surface (W/m/K)
        lambda_int_plus: Thermal conductivity from the internal surface to grid point n+1 (W/m/K)
        delta_x_ext_minus: Distance from grid point n to the external surface of the control volume (m)
        delta_x_ext_plus: Distance from the external surface of the control volume to the grid point n+1 (m)
        delta_x_int_minus: Distance from grid point n to the internal surface of the control volume (m)
        delta_x_int_plus: Distance from the internal surface of the control volume to the grid point n+1 (m)
        Tw_prev: Temperature of wall at previous timestep (array, K)
        dx: Grid spacing (m)
        dt: Timestep duration (s)
        k_int: Internal heat transfer coefficient (W/m2/K)
        k_ext: External heat transfer coefficient (W/m2/K)
        T1: Temperature inside the tank (K)
        T_amb: Ambient temperature (K)
    Returns:
        Updated temperature profile of the wall (array, K)
    """
    Tw = np.copy(Tw_prev)
    N = len(Tw_prev)


# #does fact that initialize values of temp at wall create an incorrect offset? rethink about it
#     # Update for internal nodes
#     for n in range(1, N):
#         Tw[n] = ((Tw_prev[n+1]-Tw_prev[n])/(delta_x_ext_minus/lambda_ext_minus+delta_x_ext_plus/lambda_ext_plus)-(Tw_prev[n]-Tw_prev[n-1])/(delta_x_int_minus/lambda_int_minus+delta_x_int_plus/lambda_int_plus)
#                  +Tw_prev[n]*dx*rho_w_n*cp_wall_n/dt)*dt/(dx*rho_w_n*cp_wall_n)
    
#     alpha = -1/(delta_x_int_plus/lambda_int_plus)
#     beta = -1/(delta_x_ext_minus/lambda_ext_minus)

#     # Update for internal boundary (n = 0)
#     Tw[0] = (k_int*T1-alpha*Tw[1])/(k_int-alpha)

    
#     # Update for external boundary (n = N-1)
#     Tw[-1] = (k_ext*T_amb-beta*Tw[-2])/(k_ext-beta)

    for n in range(1, N_prime):
        Tw[n] = ((Tw_prev[n+1]-Tw_prev[n])/(dx/lambda_liner)-(Tw_prev[n]-Tw_prev[n-1])/(dx/lambda_liner)
                 +Tw_prev[n]*dx*rho_liner*cp_liner/dt)*dt/(dx*rho_liner*cp_liner)
        
    Tw[N_prime] = ((Tw_prev[n+1]-Tw_prev[n])/(dx/lambda_CFRP)-(Tw_prev[n]-Tw_prev[n-1])/(dx/lambda_liner)
                  +Tw_prev[n]*dx*rho_w_n*cp_wall_n/dt)*dt/(dx*rho_w_n*cp_wall_n)
    
    for n in range(N_prime+1, N):
        Tw[n] = ((Tw_prev[n+1]-Tw_prev[n])/(dx/lambda_CFRP)-(Tw_prev[n]-Tw_prev[n-1])/(dx/lambda_CFRP)
                 +Tw_prev[n]*dx*rho_CFRP*cp_CFRP/dt)*dt/(dx*rho_CFRP*cp_CFRP)
        
    alpha = -1/(dx/2/lambda_liner)
    beta = -1/(dx/2/lambda_CFRP)

    # Update for internal boundary (n = 0)
    Tw[0] = (k_int*T1-alpha*Tw[1])/(k_int-alpha)

    # Update for external boundary (n = N-1)
    Tw[-1] = (k_ext*T_amb-beta*Tw[-2])/(k_ext-beta)

    return Tw