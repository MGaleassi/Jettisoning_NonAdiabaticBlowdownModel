import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from F_inside_tank import *
from F_tank_wall import *
from F_inside_oriface import *
from N_checker import calculate_if_node_at_material_interface_int,dx_finder


# Main Simulation Function
def main_simulation(total_time,dt,P1, V, b, gamma,T_w_0, R_g, N, d_liner, 
                    d_CFRP, CD, g, beta,P_amb,D2_orifice,cp_gas,D_int, mu_g, 
                    lambda_g, rho_air, V_air, D_ext, mu_air, cp_air, lambda_air, 
                    rho_w_n, cp_wall_n, rho_liner, rho_CFRP, cp_liner, cp_CFRP,
                    A_int, T_amb, lambda_liner, lambda_CFRP):
    """
    Main function to run the blowdown simulation based on the provided initial conditions.
    Args:
        P1_0: Initial pressure inside the tank (Pa)
        T1_0: Initial temperature inside the tank (K)
        V: Volume of the tank (m3)
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
        cp_gas: Specific heat capacity of the gas (J/kg/K)
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

    rho1 = calculate_density_abel_noble_rho1_initial(T1, P1, b, R_g)
    m = V*rho1
    
    U = calculate_internal_energy(P1, V, m, b, gamma)
    Tw = np.ones(int(1 / dx)) * T_w_0  # Initialize wall temperature profile
    Tw_prev = Tw  
    results = []

    N_prime = calculate_if_node_at_material_interface_int(N, d_liner, d_CFRP)
    dx = dx_finder(N_prime, d_liner)
    
    for i in range(time_steps):
        #Current state
        #Calculate parameters in oriface
        rho2 = calculate_density_orifice_rho2(rho1, b, gamma)
        T2 = calculate_temperature_energy_conservation_T2(rho2, T1, b, gamma)
        P2 = calculate_pressure_abel_noble_P2(T2, rho2, b, R_g)
        u2 = calculate_velocity(T2, gamma, R_g, rho2, b)

        #Calculate parameters in Notional Nozzle
        T3 = calculate_temperature_energy_conservation_T3(P2, rho2, T2, b, R_g, gamma)
        rho3 = calculate_density_abel_noble_rho3(T3, P_amb, b, R_g)
        u3 = calculate_velocity(T3, gamma, R_g, rho3, b)
        D3_nozzle = calculate_notional_nozzle_diameter(D2_orifice, CD, rho2, u2, rho3, u3)

        #Calculate massflow
        mass_flow_rate = calculate_mass_flow_rate(rho3, u3, D3_nozzle)

        # Calculate Rayleigh number and Nusselt number for internal convection
        Ra_int = calculate_rayleigh_number_internal(g, beta, T1, Tw[0], cp_gas, rho1, D_int, mu_g, lambda_g)
        Nu_Din = calculate_nusselt_number_internal(Ra_int)
        k_int = calculate_internal_surface_heat_transfer_coefficient(Nu_Din, lambda_g, D_int)
        
        # Calculate external convection properties (might be zero if Vair is zero)
        Re_Dext = calculate_reynolds_number_external(rho_air, V_air, D_ext, mu_air)
        Pr_air = calculate_prandtl_number(mu_air, cp_air, lambda_air)
        Nu_Dext = calculate_nusselt_number_external(Re_Dext, Pr_air)
        k_ext = calculate_external_surface_heat_transfer_coefficient(Nu_Dext, lambda_air, D_ext)
        
        results.append({
            "time": i * dt,
            "P1": P1,
            "T1": T1,
            "m": m,
            "rho1": rho1,
            "u2": u2,
            "D3_nozzle": D3_nozzle,
            "mass_flow_rate": mass_flow_rate,
            "T_w_internal": Tw_prev[0],
            "T_w_external": Tw_prev[-1],
            "Tw_profile": Tw_prev
        })
        
        #Updates
        #Update mass
        m -= mass_flow_rate*dt
        
        # Update wall temperature using finite difference solver
        Tw = heat_conduction_solver(rho_w_n, cp_wall_n, rho_liner, rho_CFRP, cp_liner, cp_CFRP, Tw_prev, dx, dt, k_int, k_ext, T1, T_amb, N_prime, lambda_liner, lambda_CFRP)  
        Tw_prev = Tw 
        
        # Calculate heat transfer rate from internal wall to gas
        Q_dot = calculate_heat_transfer(k_int, A_int, Tw[0], T1)
        
        # Update internal energy 
        dU_dt = Q_dot-cp_gas*T1*mass_flow_rate  
        U += dU_dt * dt

        # Recalculate parameters in the tank
        P1 = calculate_internal_pressure_P1_from_internal_energy(U, V, m, b, gamma)
        rho1 = m/V
        T1 = calculate_temperature_abel_noble_T1(P1, rho1, b, R_g)        
        
    return results