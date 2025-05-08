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


system = DoublePendulumSystem(1, 1, 1, 1)

t_start = 0
t_end = 1000 # Intergration Time
t_span = [t_start, t_end] 
t_arr = np.linspace(t_start, t_end, 10000)

# Defining the range of ICs for varying phi2 from 0 to pi with 314 points inbetween them.

q_init_array = np.array([[0, 0, i, 0] for i in np.linspace(-1e-6, np.pi-1e-6, 314)])

# Create an event to count the number of time phi1 passes through zero in the positive direction

poincare_section_event = create_poincare_section_event(tracker_index=0, direction=1)

n_cores = os.cpu_count() - 1

def solve_ivp_task(q_init):
    return solve_ivp(system.deriv, t_span, q_init,
                            t_eval = t_arr,
                            events=[poincare_section_event],
                            rtol = 1e-9, atol = 1e-9)


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        solutions = list(executor.map(solve_ivp_task, q_init_array))

    filename = "bifurcation_varying_phi2"

    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(solutions, f)