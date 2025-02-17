import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from VariablesPaper2 import *
from F_inside_tank import *
from F_tank_wall import *
from F_inside_oriface import *
from F_main import *

# Running the Simulation
if __name__ == "__main__":
    results = main_simulation(total_time, dt, P1, V, b, gamma, T_w_0, R_g, N, 
                              d_liner, d_CFRP, CD, g, beta, P_amb, D2_orifice,
                              cp_gas,D_int, mu_g, lambda_g, rho_air, V_air, 
                              D_ext, mu_air, cp_air, lambda_air, rho_w_n, 
                              cp_wall_n, rho_liner, rho_CFRP, cp_liner, cp_CFRP,
                              A_int,T_amb, lambda_liner, lambda_CFRP)
    
    # Extract data from the simulation results for plotting
    time = [res['time'] for res in results]
    pressure = [res['P1'] / 1e5 for res in results]  # Convert from Pa to bar for pressure
    temperature = [res['T1'] for res in results]
    mass = [res['m'] for res in results]
    density = [res['rho1'] for res in results]
    velocity_throat = [res['u2'] for res in results]
    D3_lst = [res['D3_nozzle'] for res in results]
    mass_flow_lst = [res['mass_flow_rate'] for res in results]
    T_w_internal = [res['T_w_internal'] for res in results]
    T_w_external = [res['T_w_external'] for res in results]
    Tw_profile = [res['Tw_profile'] for res in results]


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

    # Plotting the mass over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, mass, label="Mass of gas", color="blue", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Mass (kg)")
    plt.title("Mass over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the density over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, density, label="Density of gas", color="green", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Density (kg/m³)")
    plt.title("Density over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the velocity at the throat over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, velocity_throat, label="Velocity at the throat", color="red", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity at the Throat over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the nozzle diameter over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, D3_lst, label="Nozzle diameter", color="purple", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Diameter (m)")
    plt.title("Nozzle Diameter over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the mass flow rate over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, mass_flow_lst, label="Mass flow rate", color="orange", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Mass Flow Rate (kg/s)")
    plt.title("Mass Flow Rate over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the internal wall temperature over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, T_w_internal, label="Internal wall temperature", color="cyan", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("Internal Wall Temperature over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the external wall temperature over time
    plt.figure(figsize=(10, 5))
    plt.plot(time, T_w_external, label="External wall temperature", color="magenta", linestyle="solid", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (K)")
    plt.title("External Wall Temperature over Time during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting the wall temperature profile over time
    plt.figure(figsize=(10, 5))
    for profile in Tw_profile:
        plt.plot(profile, label="Wall temperature profile")
    plt.xlabel("Grid Points")
    plt.ylabel("Temperature (K)")
    plt.title("Wall Temperature Profile during Hydrogen Gas Discharge")
    plt.legend()
    plt.grid(True)
    plt.show()