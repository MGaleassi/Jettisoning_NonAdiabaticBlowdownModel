import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from VariablesPaper2 import *

#Abel-Noble accounts for behaviour of real gasses at high pressures by accounting for finite size of atoms

# Constants
GRAVITY = 9.81  # m/s^2
GAS_CONSTANT = 4124  # Hydrogen specific, m2 s2 K^-1
AIR_SPECIFIC_HEAT = 1005  # J/kg/K for air

# Functions Section

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
def calculate_internal_energy(P1, V, m1, b, gamma):#kinda uselful only once
    """
    Calculates the internal energy of the gas inside the tank.
    Args:
        P1: Pressure inside the tank (Pa)
        V: Volume of the tank (m3)
        m1: Mass of gas inside the tank (kg)
        b: Co-volume constant (m3/kg)
        gamma: Specific heat ratio (-)
    Returns:
        Internal energy in the tank (J)
    """
    return (P1 * (V - m1 * b)) / (gamma - 1)

# Function to calculate P1 from internal energy in the tank
def calculate_internal_pressure_P1_from_internal_energy(U, V, m1, b, gamma):
    """
    Calculates the internal energy of the gas inside the tank.
    Args:
        U: Internal energy in the tank (J)
        V: Volume of the tank (m3)
        m1: Mass of gas inside the tank (kg)
        b: Co-volume constant (m3/kg)
        gamma: Specific heat ratio (-)
    Returns:
        Pressure inside the tank (Pa)
    """
    return (gamma - 1)*U/(V-m1*b)

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
        Temperature inside the tank (K)
    """
    return P_amb / (P_amb*b + R_g*T3)

# Function to calculate the diameter of the notional nozzle (D3_nozzle)
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
def calculate_mass_flow_rate(rho3, u3, D3_nozzle):
    """
    Calculate the mass flow rate at the notional nozzle.
    Args:
        rho3: Density at the notional nozzle (kg/m3)
        u3: Velocity at the notional nozzle (m/s)
        D3_nozzle: Diameter of the notional nozzle (m)
    Returns:
        Mass flow rate (kg/s)
    """
    return rho3 * u3 * np.pi * (D3_nozzle ** 2) / 4

# Function to calculate Rayleigh number inside the tank (Ra_int)
def calculate_rayleigh_number_internal(g, beta, T1, T_w_int, cp_g, rho1, D_int, mu_g, lambda_g):
    """
    Calculate Rayleigh number for internal gas convection inside the tank.
    Args:
        g: Acceleration due to gravity (m/s^2)
        beta: Thermal expansion coefficient of the gas (1/K)
        T1: Temperature inside the tank (K)
        T_w_int: Temperature of the inside surface of the tank wall (K)
        cp_g: Specific heat capacity of the gas (J/kg/K)
        rho1: Density of the gas inside the tank (kg/m3)
        D_int: Internal diameter of the tank (m)
        mu_g: Dynamic viscosity of the gas (Pa.s)
        lambda_g: Thermal conductivity of the gas (W/m/K)
    Returns:
        Rayleigh number (dimensionless)
    """
    return (g * beta * abs(T1 - T_w_int) * cp_g * (rho1 ** 2) * (D_int ** 3)) / (mu_g * lambda_g)


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

# Function to calculate if discretization number is acceptable (N_prime)
def calculate_if_node_at_material_interface_int(Nu_Dext, lambda_air, D_ext):
    """
    Calculate if the node at the material interface in the wall of the tank is an integer and what value it has.
    Args:
        Nu_Dext: Nusselt number for external convection (dimensionless)
        lambda_air: Thermal conductivity of the ambient air (W/m·K)
        D_ext: External diameter of the tank (m)
    Returns:
        Heat transfer coefficient of the external surface of the wall (W/m²·K)
    """
    return (lambda_air*Nu_Dext)/D_ext

# Finite difference solver for 1D unsteady heat conduction within tank wall
def heat_conduction_solver(rho_wall, cp_wall, lambda_wall, Tw_prev, dx, dt, k_int, k_ext, T1, T_amb):
    """
    Solves the unsteady heat conduction equation within the tank wall using the finite difference method.
    Args:
        rho_wall: Density of the wall material (kg/m3)
        cp_wall: Specific heat capacity of the wall material (J/kg/K)
        lambda_wall: Thermal conductivity of the wall material (W/m/K)
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

    # Update for internal nodes
    for n in range(1, N - 1):
        Tw[n] = 
    
    # Update for internal boundary (n = 0)
    Tw_new[0] = Tw_prev[0] + (lambda_wall / (rho_wall * cp_wall)) * dt * (
        (Tw_prev[1] - Tw_prev[0]) / (dx ** 2) - (dx / (lambda_wall / k_int)) * (T1 - Tw_prev[0])
    )
    
    # Update for external boundary (n = N-1)
    Tw_new[-1] = Tw_prev[-1] + (lambda_wall / (rho_wall * cp_wall)) * dt * (
        (Tw_prev[-2] - Tw_prev[-1]) / (dx ** 2) - (dx / (lambda_wall / k_ext)) * (Tw_prev[-1] - T_amb)
    )
    
    
    
    return Tw_new

# Main Simulation Function
def main_simulation(P1_0, T1_0, V, m1_0, gamma, b, R_g, k_int, A_int, Tw_int_0, dt, total_time, D2_orifice, CD, D_int, D_ext, cp_g, mu_g, lambda_g, beta, T_amb, L_ext, nu_air, alpha_air, Pr_air, V_air, lambda_wall, rho_wall, cp_wall, dx, lambda_CFRP, lambda_liner, rho_CFRP, rho_liner, cp_CFRP, cp_liner):
    """
    Main function to run the blowdown simulation based on the provided initial conditions.
    Args:
        P1_0: Initial pressure inside the tank (Pa)
        T1_0: Initial temperature inside the tank (K)
        V: Volume of the tank (m3)
        m1_0: Initial mass of gas inside the tank (kg)
        gamma: Specific heat ratio (-)
        b: Co-volume constant (m3/kg)
        R_g: Gas constant (m2 s2 K^-1)
        k_int: Heat transfer coefficient (W/m2/K)
        A_int: Internal surface area of the tank (m2)
        Tw_int_0: Initial temperature of internal wall (K)
        dt: Time step (s)
        total_time: Total simulation time (s)
        D2_orifice: Diameter of the orifice (m)
        CD: Discharge coefficient (-)
        D_int: Internal diameter of the tank (m)
        D_ext: External diameter of the tank (m)
        cp_g: Specific heat capacity of the gas (J/kg/K)
        mu_g: Dynamic viscosity of the gas (Pa.s)
        lambda_g: Thermal conductivity of the gas (W/m/K)
        beta: Thermal expansion coefficient of the gas (1/K)
        T_amb: Ambient temperature (K)
        L_ext: Characteristic length of the tank (m)
        nu_air: Kinematic viscosity of air (m2/s)
        alpha_air: Thermal diffusivity of air (m2/s)
        Pr_air: Prandtl number of air (-)
        V_air: Velocity of air for external forced convection (m/s)
        lambda_wall: Thermal conductivity of wall material (W/m/K)
        rho_wall: Density of wall material (kg/m3)
        cp_wall: Specific heat capacity of wall material (J/kg/K)
        dx: Grid spacing for wall discretization (m)
        lambda_CFRP: Thermal conductivity of CFRP (W/m/K)
        lambda_liner: Thermal conductivity of liner (W/m/K)
        rho_CFRP: Density of CFRP (kg/m3)
        rho_liner: Density of liner (kg/m3)
        cp_CFRP: Specific heat capacity of CFRP (J/kg/K)
        cp_liner: Specific heat capacity of liner (J/kg/K)
    Returns:
        Time evolution of pressure, temperature, mass inside the tank, and mass flow rate.
    """
    time_steps = int(total_time / dt)
    P1, T1, m1 = P1_0, T1_0, m1_0
    U = calculate_internal_energy(P1, V, m1, b, gamma)
    Tw_prev = np.ones(int(1 / dx)) * Tw_int_0  # Initialize wall temperature profile
    results = []
    
    for i in range(time_steps):
        # Calculate Rayleigh number and Nusselt number for internal convection
        rho1 = m1 / V
        Ra_int = calculate_rayleigh_number_internal(GRAVITY, beta, T1, Tw_prev[0], cp_g, rho1, D_int, mu_g, lambda_g)
        Nu_Din = calculate_nusselt_number_internal(D_int, Ra_int)
        k_int = calculate_internal_surface_heat_transfer_coefficient(Nu_Din, lambda_g, D_int)
        
        # Calculate external convection properties
        Re_Dext = calculate_reynolds_number_external(rho_air, V_air, D_ext, mu_air)
        Pr_air = calculate_prandtl_number(mu_air, cp_air, lambda_air)
        Nu_Dext = calculate_nusselt_number_external(Re_Dext, Pr_air)
        k_ext = calculate_external_surface_heat_transfer_coefficient(Nu_Dext, lambda_air, D_ext)
        
        # Update wall temperature using finite difference solver
        Tw_new = heat_conduction_solver(rho_wall, cp_wall, lambda_wall, Tw_prev, dx, dt, k_int, k_ext, T1, T_amb)
        Tw_prev = Tw_new  # Update wall temperature profile for next iteration
        
        # Calculate heat transfer rate from internal wall to gas
        Q_dot = calculate_heat_transfer(k_int, A_int, Tw_prev[0], T1)
        
        # Update internal energy and mass
        dU_dt = Q_dot  # Assuming no other energy inputs or losses
        U += dU_dt * dt
        m1 -= Q_dot * dt / (gamma * R_g * T1)
        
        # Update pressure and temperature
        T1 = calculate_temperature_abel_noble_T1(P1, rho1, b, R_g)
        P1 = rho1 * R_g * T1 / (1 - b * rho1)# this is incorrect P1 is updated using U
        
        # Calculate properties at orifice and notional nozzle
        rho2 = calculate_density_orifice_rho2(rho1, b, gamma)
        T2 = calculate_temperature_energy_conservation_T2(rho2, T1, b, gamma)
        u2 = calculate_velocity(T2, gamma, R_g, rho2)
        
        # Notional nozzle calculations
        T3 = calculate_temperature_energy_conservation_T3(P2, rho2, T2, b, R_g, gamma)
        rho3 = P1 / (P1 * b + R_g * T3)
        u3 = calculate_velocity(T3, gamma, R_g, rho3)
        D3_nozzle = calculate_notional_nozzle_diameter(D2_orifice, CD, rho2, u2, rho3, u3)
        m_dot = calculate_mass_flow_rate(rho3, u3, D3_nozzle)
        
        results.append({
            "time": i * dt,
            "P1": P1,
            "T1": T1,
            "m1": m1,
            "rho1": rho1,
            "rho2": rho2,
            "T2": T2,
            "u2": u2,
            "T3": T3,
            "rho3": rho3,
            "u3": u3,
            "D3_nozzle": D3_nozzle,
            "m_dot": m_dot,
            "k_int": k_int,
            "k_ext": k_ext,
            "T_w_internal": Tw_prev[0],
            "T_w_external": Tw_prev[-1],
            "Tw_profile": Tw_prev
        })
    
    return results

# Running the Simulation
if __name__ == "__main__":
    results = main_simulation(
        P1_0=7.0e7,  # Pa
        T1_0=293,  # K
        V=0.019,  # m3 (19 liters)
        m1_0=1.67,  # kg
        gamma=1.66,  # -
        b=2.67e-3,  # m3/kg
        R_g=GAS_CONSTANT,  # m2 s2 K^-1
        k_int=0.104,  # W/m2/K (example value)
        A_int=1.0,  # m2 (example value)
        Tw_int_0=293,  # K
        dt=0.1,  # s
        total_time=300,  # s
        D2_orifice=0.01,  # m (example orifice diameter)
        CD=0.9,  # Discharge coefficient (example value)
        D_int=0.18,  # m (example internal diameter of the tank)
        D_ext=0.2,  # m (example external diameter of the tank)
        cp_g=14300,  # J/kg/K (specific heat capacity for hydrogen)
        mu_g=8.76e-6,  # Pa.s (dynamic viscosity for hydrogen)
        lambda_g=0.1805,  # W/m/K (thermal conductivity for hydrogen)
        beta=1 / 293,  # 1/K (thermal expansion coefficient for ideal gas)
        T_amb=293,  # K (ambient temperature)
        L_ext=0.5,  # m (characteristic length of the tank)
        nu_air=1.5e-5,  # m2/s (kinematic viscosity of air)
        alpha_air=2.2e-5,  # m2/s (thermal diffusivity of air)
        Pr_air=0.71,  # Prandtl number of air
        V_air=2.0,  # m/s (velocity of air for external forced convection)
        lambda_wall=0.5,  # W/m/K (example value for wall material)
        rho_wall=7850,  # kg/m3 (density of wall material, e.g., steel)
        cp_wall=500,  # J/kg/K (specific heat capacity of wall material)
        dx=0.01,  # m (grid spacing for wall discretization)
        lambda_CFRP=0.5,  # W/m/K (example value for CFRP material)
        lambda_liner=0.2,  # W/m/K (example value for liner material)
        rho_CFRP=1600,  # kg/m3 (density of CFRP material)
        rho_liner=1000,  # kg/m3 (density of liner material)
        cp_CFRP=900,  # J/kg/K (specific heat capacity of CFRP material)
        cp_liner=700   # J/kg/K (specific heat capacity of liner material)
    )
    
    # Extract data from the simulation results for plotting
    time = [res['time'] for res in results]
    pressure = [res['P1'] / 1e5 for res in results]  # Convert from Pa to bar for pressure
    temperature = [res['T1'] for res in results]

    # Plotting the pressure over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, pressure, label="Simulated gas pressure - non-adiabatic model", color="black", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (bar)")
    plt.title("Pressure over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the temperature over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, temperature, label="Simulated gas temperature - non-adiabatic model", color="black", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Temperature over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()
