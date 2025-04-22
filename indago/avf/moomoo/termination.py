import numpy as np

from pymoo.util.normalization import normalize
from pymoo.termination.delta import DeltaToleranceTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.hv import HyperVolume


def calc_delta(a, b):
    return np.max(np.abs((a - b)))


def calc_delta_norm(a, b, norm):
    return np.max(np.abs((a - b) / norm))


class SingleObjectiveSpaceTermination(DeltaToleranceTermination):

    def __init__(self, tol=1e-6, n_last=0, n_gen=50, only_feas=True, **kwargs) -> None:
        super().__init__(tol, **kwargs)
        self.only_feas = only_feas
        self.n_last = n_last
        self.n_gen = n_gen
        self.n_counter = 0

    def _delta(self, prev, current):
        if prev == np.inf or current == np.inf:
            return np.inf
        else:
            val = max(0, prev - current)
            if val <= self.tol and self.n_counter >= self.n_last:
                return val
            elif val <= self.tol:
                self.n_counter += 1
                return np.inf
            else:
                self.n_counter = 0
                return np.inf

    def _data(self, algorithm):
        if algorithm.n_gen == self.n_gen:
            algorithm.termination.force_termination = True
            return np.inf  # Force termination
        
        opt = algorithm.opt
        f = opt.get("f")

        if self.only_feas:
            f = f[opt.get("feas")]

        if len(f) > 0:
            return f.min()
        else:
            return np.inf


class MultiObjectiveHypervolumeTermination(DeltaToleranceTermination):
    
    def __init__(self, ref_point, tol=1e-6, n_last=10, n_gen=100):
        """
        Termination based on hypervolume improvement for multi-objective optimization.

        Parameters:
        - ref_point (array-like): Reference point for hypervolume calculation.
        - tol (float): Minimum required improvement in hypervolume to continue.
        - n_last (int): Number of generations with minimal improvement before stopping.
        - n_max_gen (int): Maximum number of generations before termination.
        """
        super().__init__(tol=tol)
        self.ref_point = np.array(ref_point)
        self.n_last = n_last
        self.n_gen = n_gen
        self.hv_calculator = HyperVolume(self.ref_point)
        self.prev_hv = None
        # self.stagnation = 0
        # self.stagnent = False
        self.n_counter = 0  # Tracks generations with minimal improvement

    def _delta(self, prev, current):
        if prev == np.inf or current == np.inf:
            return np.inf
        else:
            improvement = current - prev  # Hypervolume should increase
            if improvement <= self.tol and self.n_counter >= self.n_last:
                return improvement
            elif improvement <= self.tol:
                self.n_counter += 1
                # if self.stagnation <= 1:
                #     self.stagnent = True
                return np.inf
            else:
                self.n_counter = 0
                # if self.stagnent:
                #     self.stagnent = False
                #     self.stagnation += 1
                return np.inf

    def _data(self, algorithm):
        if algorithm.n_gen >= self.n_gen:
            algorithm.termination.force_termination = True
            return np.inf  # Force termination
        
        # algorithm.stagnation = self.stagnation
        # algorithm.stagnent = self.stagnent
        # Extract non-dominated solutions
        opt = algorithm.opt
        F = opt.get("F")  # Objective values
        feas = opt.get("feas")

        # Filter only feasible solutions
        if feas is not None:
            F = F[feas]

        # Ensure F is not empty
        if len(F) > 0:
            nds = NonDominatedSorting()
            front_indices = nds.do(F, only_non_dominated_front=True)[0]
            front = F[front_indices]  # Properly extract non-dominated solutions

            # Ensure front is a 2D array
            front = np.atleast_2d(front)

            # Compute hypervolume
            return self.hv_calculator.compute(front)
        else:
            return np.inf
