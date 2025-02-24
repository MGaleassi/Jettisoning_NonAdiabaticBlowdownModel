#all variables

# A = None # Surface area of the metal plate on top of the thermocouple (m²)
A_int = 721.91*10**-3 # Internal surface area of the tank (m²)
b = 0.01316 # Co-volume constant of the gas for the Abel-Noble equation (m³/kg)
CD = None # Discharge coefficient (dimensionless)
cp_air = 1.01*10**3 # Specific heat capacity of air (J/kg·K)
cp_CFRP = None # Specific heat capacity of CFRP (J/kg·K)
cp_liner = None # Specific heat capacity of the liner (J/kg·K)
cp_plate = None # Specific heat capacity of the metal plate on top of the thermocouple (J/kg·K)
cp_gas = None # Specific heat capacity of the inside gas at constant pressure (J/kg·K)
cv_gas = None # Specific heat capacity of the inside gas at constant volume (J/kg·K)
cp_wall_n = None # Specific heat capacity of the wall (Liner or CFRP) at a grid point (J/kg·K)
d_liner = 26.9*10**-3 # Thickness of the liner of the tank(m)
d_CFRP = 65.4*10**-3 # Thickness of the CFRP of the tank(m)
D2_orifice = 2*10**-3 # Diameter of the orifice (m)
# D3_nozzle = None # Diameter of the effective nozzle (m)
D_ext = None # External diameter of the tank (m)
D_int = None # Internal diameter of the tank (m)
D_plate = None # Diameter of the metal plate on top of the thermocouple (m)
g = 9.81  # Acceleration due to gravity (m/s²)
h_out = None # Enthalpy going out (J/kg)
iteration = 0 # Iteration number (integer)
k_ext = None # Heat transfer coefficient of the external surface of the wall (W/m²·K)
k_int = None # Heat transfer coefficient of the internal surface of the wall (W/m²·K)
k_plate = None # Heat transfer coefficient of the metal plate on top of the thermocouple (W/m²·K)
m_vessel = None # Mass in the vessel (kg)
n = 0 # Grid point number (integer)
L_plate = None # Thickness of the metal plate on top of the thermocouple (m)
Mg = None # Molar mass of the inside gas (g/mol)
Nu_air = None # Nusselt number of the air (dimensionless)
Nu_gas = None # Nusselt number of the inside gas (dimensionless)
Nu_plate = None # Nusselt number of the metal plate on top of the thermocouple (dimensionless)
P1 = 70*10**6 # Pressure of the gas inside the tank (Pa)
# P2 = None # Pressure of the gas at the orifice (Pa)
P_amb = 1.01*10**5 # Ambient pressure (Pa); Can probably also make variable with altitude as different scenarios
Pr_air = None # Prandtl number of the air (dimensionless)
# Q = None # Heat to the system due to surrounding (J)
# Ra_int = None # Rayleigh number of the inside gas (dimensionless)
# Ra_plate = None # Rayleigh number of the metal plate on top of the thermocouple (dimensionless)
# Re_air = None # Reynolds number of the air (dimensionless)
R_g = None # Gas constant (m²/s²·K)
S = None # Source term (W/m³·s)
t = None # Time (s)
T1 = None # Temperature of the gas inside the tank (K)
T2 = None # Temperature of the gas at the orifice (K)
T3 = None # Temperature at the notional nozzle (K)
T_amb = 293 # Ambient temperature (K); consider making variabke with altitude
T_plate = None # Temperature of the metal plate on top of the thermocouple (K)
T_w_ext = None # Temperature of the external surface of the tank (K)
T_w_int = None # Temperature of the internal surface of the tank (K)
T_w_n = None # Temperature of the wall at a grid-point (K)
T_w_N = None # Temperature of the last grid point before the external surface (K)
T_w_0 = 293.15 # Initialize wall temperature
U = None # Internal energy in the tank (J)
u2 = None # Velocity in the orifice (m/s)
u3 = None # Velocity at the notional nozzle (m/s)
V = 0.051 # Volume of the tank (m³)
V_air = 0 # Velocity of surrounding air (m/s)
V_plate = None # Volume of the metal plate on top of the thermocouple (m³)
delta_x_ext_minus = None # Distance from grid point n to the external surface of the control volume (m)
delta_x_ext_plus = None # Distance from the external surface of the control volume to the grid point n+1 (m)
delta_x_int_minus = None # Distance from the internal surface of the control volume to the grid point n-1 (m)
delta_x_int_plus = None # Distance from grid point n to the internal surface of the control volume (m)
lambda_ext_minus = None # Thermal conductivity of the wall between grid point n and the external surface of the control volume (W/m·K)
lambda_ext_plus = None # Thermal conductivity of the wall between the external surface of the control volume and grid point n+1 (W/m·K)
lambda_int_minus = None # Thermal conductivity of the wall between the internal surface of the control volume and grid point n-1 (W/m·K)
lambda_int_plus = None # Thermal conductivity of the wall between grid point n and the internal surface of the control volume (W/m·K)
mass_flow_rate = None # Mass flow rate (kg/s)
m = None #Mass of gas in tank (kg)
delta_x = None # Length of control volumes (m)
beta = None # Thermal expansion coefficient of the gas (K⁻¹)
gamma = 1.4 # Ratio of specific heat capacity (dimensionless)
lambda_air = None # Thermal conductivity of the ambient air (W/m·K)
lambda_CFRP = None # Thermal conductivity of CFRP (W/m·K)
lambda_g = None # Thermal conductivity of the gas (W/m·K)
lambda_liner = None # Thermal conductivity of the liner (W/m·K)
mu_air = None # Dynamic viscosity of the air (Pa·s)
mu_g = None # Dynamic viscosity of the gas (Pa·s)
rho1 = None # Density of the gas inside the tank (kg/m³)
rho2 = None # Density of the gas at the orifice (kg/m³)
rho3 = None # Density at the notional nozzle (kg/m³)
rho_air = None # Density of air (kg/m³)
rho_CFRP = None # Density of CFRP (kg/m³)
rho_liner = None # Density of liner (kg/m³)
rho_plate = None # Density of the metal plate on top of the thermocouple (kg/m³)
rho_w_n = None # Density of the wall (Liner or CFRP) at a grid point n (kg/m³)

total_time = 900 #total simulation time (s)
dt = 0.01 #simulation time step
