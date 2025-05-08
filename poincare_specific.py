import numpy as np
from scipy.integrate import solve_ivp
import concurrent.futures
import os
import pickle

class DoublePendulumSystem:
    def __init__(self, M1, M2, L1, L2):
        self.M1 = M1
        self.M2 = M2
        self.L1 = L1
        self.L2 = L2
        self.grav = 9.81
    
    def solve_ddphi1(self, phi1, dphi1, phi2, dphi2, M1, M2, L1, L2, grav):
        numerator = (
            -grav * (2 * M1 + M2) * np.sin(phi1)
            - M2 * grav * np.sin(phi1 - 2 * phi2)
            - 2 * np.sin(phi1 - phi2) * M2 * (dphi2**2 * L2 + dphi1**2 * L1 * np.cos(phi1 - phi2))
        )
        
        denominator = L1 * (2 * M1 + M2 - M2 * np.cos(2 * phi1 - 2 * phi2))

        return numerator / denominator

    def solve_ddphi2(self, phi1, dphi1, phi2, dphi2, M1, M2, L1, L2, grav):
        numerator = (
            2 * np.sin(phi1 - phi2)
            * (
                dphi1**2 * L1 * (M1 + M2)
                + grav * (M1 + M2) * np.cos(phi1)
                + dphi2**2 * L2 * M2 * np.cos(phi1 - phi2)
            )
        )

        denominator = L2 * (2 * M1 + M2 - M2 * np.cos(2 * phi1 - 2 * phi2))

        return numerator / denominator

    def deriv(self, t_now, q_now):
        M1 = self.M1
        M2 = self.M2
        L1 = self.L1
        L2 = self.L2
        grav = self.grav
    
        # Unpack the phi values and their first derivatives
        phi1, dphi1, phi2, dphi2 = q_now

        ddphi1 = self.solve_ddphi1(phi1, dphi1, phi2, dphi2, M1, M2, L1, L2, grav)
        ddphi2 = self.solve_ddphi2(phi1, dphi1, phi2, dphi2, M1, M2, L1, L2, grav)

        d_array = np.array([dphi1, ddphi1, dphi2, ddphi2])

        return d_array

def convert_T_to_velocity(T, M, direction=1):
    speed = np.sqrt(2*T/M)

    if direction == 1: velocity = speed
    elif direction == -1: velocity = -speed
    else: raise ValueError("Direction must be 1 or -1 for positive and negative directions")

    return velocity

def create_poincare_section_event(tracker_index, direction=1):
    """"Takes: the index of the variable to be tracked for zero crossings inside solve_ivp;
    and the direction of the sign change with 1 ~ negative to positive, 
    -1 ~ negative to positive, 0 ~ both. Returns the event function that
    should be supplied in the solve_ivp call."""

    def poincare_section_event(t_now, q_now):
        return q_now[tracker_index]

    poincare_section_event.terminal = False             # Keep solving after event
    poincare_section_event.direction = direction         

    return poincare_section_event

def solve_dphi2_given_dphi1(T_target, q, system):
    """"Solve for dphi1 given initial phi1, phi2, dphi2 and target value of KE, T. This is
    to be used to find possible combinations for of pphis for fixed values of T, to construct
    Poincare sections and bifurcation diagrams given fixed energy values."""

    phi1, dphi1, phi2, dphi2 = q

    M1 = system.M1
    M2 = system.M2
    L1 = system.L1
    L2 = system.L2
    
    A = 0.5 * (M1 + M2) * L1**2
    B = M2 * L1 * L2 * np.cos(phi1 - phi2)
    C = 0.5 * M2 * L2**2

    # Coefficients for quadratic in dphi2
    a = C
    b = B * dphi1
    c = A * dphi1**2 - T_target

    discriminant = b**2 - 4*a*c

    if discriminant < 0:
        return []  # No real solutions
    elif discriminant == 0:
        solution = (-b) / (2*a)
        return [solution, -solution]
    else:
        sqrt_discriminant = np.sqrt(discriminant)
        solution_1 = (-b + sqrt_discriminant) / (2*a)
        solution_2 = (-b - sqrt_discriminant) / (2*a)
        return [solution_1, solution_2]
    

system = DoublePendulumSystem(1, 1, 1, 1)

t_start = 0
t_end = 3000
t_span = [t_start, t_end] 
t_arr = np.linspace(t_start, t_end, 10000)

# Define 3 specific angles of release of phi2 to explore

interesting_phi2_array = [0, 2.2, 3.14]

q_init_array_array = []

# For each angle of release, create combinations of different intial velocities dphi1, dphi2 conserving the KE = 0.1 J

for interesting_phi2 in interesting_phi2_array:

    phi2 = interesting_phi2
    phi1 = 0

    T_init = 0.1
    dphi1_max = convert_T_to_velocity(T_init, system.M1, 1)/system.L1

    dphi1_array = np.linspace(0, dphi1_max, 50)

    q_init_array = []

    for dphi1 in dphi1_array:
        dphi2_array = solve_dphi2_given_dphi1(T_init, [phi1, dphi1, phi2, None], system)

        for dphi2 in dphi2_array:
            q_init_array.append(np.array([phi1, dphi1, phi2, dphi2]))

    q_init_array = np.array(q_init_array)
    q_init_array_array.append(q_init_array)

# Create an event to count the number of time phi1 passes through zero in the positive direction

poincare_section_event = create_poincare_section_event(tracker_index=0, direction=1)

n_cores = os.cpu_count() - 1


def solve_ivp_task(q_init):
    return solve_ivp(system.deriv, t_span, q_init,
                            t_eval = t_arr,
                            events=[poincare_section_event],
                            rtol = 1e-11, atol = 1e-11)


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    solutions = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        solutions.append(list(executor.map(solve_ivp_task, q_init_array_array[0])))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        solutions.append(list(executor.map(solve_ivp_task, q_init_array_array[1])))

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        solutions.append(list(executor.map(solve_ivp_task, q_init_array_array[2])))


    for i in range(len(interesting_phi2_array)):
        
        phi2 = interesting_phi2_array[i]

        filename = f'poincare_specific_phi2_{np.round(phi2/np.pi*180, 2)}_T_{T_init}'

        with open(f'{filename}.pkl', 'wb') as f:
            pickle.dump(solutions[i], f)

