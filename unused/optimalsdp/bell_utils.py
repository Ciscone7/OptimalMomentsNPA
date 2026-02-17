import numpy as np
import cvxpy as cp
import random
import itertools
import math
import os
import json
from typing import List, Dict, Optional, Any, Tuple, Union


#####FUNCTIONS#####
#BELL INEQUALITIES
def conjugate_operator_bell(s1: str) -> str:  
    """
    Parameters
    ----------
    s1 : str
        Takes a product of operators described by a string.

    Returns
    -------
    s1_dagg : str
        Returns the conjugate of the initial product of operators.
    """
    monom1_s = s1
    monom1_list = []
    while monom1_s:
        i = 1
        while i < len(monom1_s) and not monom1_s[i].isalpha():
            i += 1
        monom1_list.append(monom1_s[:i])
        monom1_s = monom1_s[i:]
    
    s1_dagg = ""
    for i in range(1, len(monom1_list)+1):
        s1_dagg += monom1_list[-i]
        
    return s1_dagg

def simplify_algebra_bell(s1: str, s2: str) -> str:
    """
    Parameters
    ----------
    s1 : str
        First operator, described by a string.
        For instance: A*B := "AB"
    s2 : str
        Second operator, described by a string.

    Returns
    -------
    string
        Returns the final operator tanking into account
        the commutation rules for projective operator and for two
        parties Alice and Bob.
    """
          
    s = s1 + s2
    subsystems = {}
    while s:
        i = 1
        while i < len(s) and not s[i].isalpha():
            i += 1
        subsystem = s[:i]
        if subsystem[0] in subsystems:
            subsystems[subsystem[0]].append(subsystem)
        else:
            subsystems[subsystem[0]] = [subsystem]
        s = s[i:]
    subsystems = dict(sorted(subsystems.items(), key=lambda x: x[0]))
    
    simplified_subsystems = {}
    for subsystem, ops in subsystems.items():
        simplified_ops = []
        for op in ops:
            if simplified_ops:
                last_op = simplified_ops[-1]
                if last_op == op:
                    continue            
                if last_op[1] == op[1] and last_op[2] != op[2]:
                    return '0'
            simplified_ops.append(op)
        simplified_subsystems[subsystem] = simplified_ops   
    
    
    final_op = []
    for subsystem, ops in simplified_subsystems.items():
        for op in ops:
            final_op.append(op)
    
    return ''.join(final_op) if final_op else ''

def simplify_linear_combination(s1: str, s_combination: str) -> str:
    s_combination_list = s_combination.split("+")
    print(s_combination_list)
    s1_times_s_comb = [simplify_algebra_bell(s1, s) for s in s_combination_list]
    
    return '+'.join(s1_times_s_comb)

def brute_force_search(starting_set: List[str], relaxation_set: List[str], bell_ineq: Dict[str, float], best_k: int) -> Tuple[float, List[str]]:
    """
    Brute-force over all subsets of size `best_k` from `relaxation_set`, appended to `starting_set`,
    and return the subset producing the maximal Bell-inequality violation.

    Args:
        starting_set (list): Base list of monomials always included.
        relaxation_set (list): List of additional monomials to choose from.
        bell_ineq:            The Bell-inequality object to test against.
        best_k (int):         Number of monomials to pick from relaxation_set.

    Returns:
        dict: {
            'best_bell_exp': float,      # highest violation found among all size-k subsets
            'best_mon_comb': list,       # the size-k subset achieving it
        }
    """
    n = len(relaxation_set)
    if not (0 <= best_k <= n):
        raise ValueError(f"best_k must be between 0 and {n}, got {best_k}")

    best_value = math.inf
    best_comb = []
    # Only consider subsets of size exactly best_k
    for comb in itertools.combinations(relaxation_set, best_k):
        candidate_set = starting_set + list(comb)
        MM = MomentMatrix_Bell(candidate_set)
        _, value = MM.inequality_violation(bell_ineq)

        if value < best_value: #we are approaching the true solution from above
            best_value = value
            best_comb = list(comb)
        
    return best_value, best_comb

def random_search(starting_set: List[str], relaxation_set: List[str], bell_ineq: Dict[str, float], number_samples: int) -> Dict[int, Dict[int, Dict[str, Union[float, List[str]]]]]:
    data_structure = {}
    for r in range(number_samples):
        data_structure[r] = {}
        for i in range(len(relaxation_set)+1):
            to_try_set = starting_set+relaxation_set[0:i]
            MM = MomentMatrix_Bell(to_try_set)
            _, value = MM.inequality_violation(bell_ineq)
            data_structure[r][i] = {
                    'bell_exp': value, #present violation of inequality 
                    'mon_comb': relaxation_set[0:i], #sequence of monomials up to this point
                }
        random.shuffle(relaxation_set)

    return data_structure

def evaluate_bell(moments: List[str], bell_expression: Dict[str, float]) -> float:
    """
    Evaluates the Bell expression with the given list of moments.

    Parameters
    ----------
    moments : list
        List of monomials (strings) forming the moment matrix.
    bell_expression : dict
        Bell expression object (dictionary of coefficients).

    Returns
    -------
    float
        The value of the Bell inequality violation.
    """
    MM = MomentMatrix_Bell(moments)
    _, value = MM.inequality_violation(bell_expression)
    return value

def _optimization_wrapper(x: np.ndarray, starting_set: List[str], adding_set: List[str], bell_expression: Dict[str, float]) -> float:
    """
    Internal wrapper that constructs the full list of moments based on 
    the binary vector x, then calls the core evaluation function.
    
    Args:
        x: Binary vector corresponding to adding_set. 
           (1 = include this element from adding_set, 0 = ignore).
        starting_set: The fixed list of moments that are always included.
        adding_set: The list of candidate moments that x selects from.
        bell_expression: The fixed Bell expression.
    """
    chosen_additions = [s for val, s in zip(x, adding_set) if val]
    total_moments = starting_set + chosen_additions

    return evaluate_bell(total_moments, bell_expression)


def load_data(file_path: str) -> np.ndarray:
    """
    Loads data from a text file where each row may have variable number of comma-separated values.
    
    Args:
        file_path (str): Path to the input file.
        
    Returns:
        np.ndarray: A numpy array containing the data. Rows with fewer elements will likely
                    cause this to be an object array or padded, but standard numpy loadtxt 
                    expects uniform rows. 
                    
                    For variable length rows, we read line by line.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                # split by comma and convert to float
                row = [float(x.strip()) for x in line.split(',')]
                data.append(row)
    
    # If rows have different lengths, we can't make a standard 2D numpy array of floats
    # usually. But for many scientific data generated here they might be uniform.
    # If they are not uniform, we can return a list of lists or an object array.
    # The user request implies variable elements, so a list of lists converted 
    # to an object array is the safest generic "read" that doesn't crash.
    # However, if they fit a rectangle, we prefer a standard array.
    
    try:
        return np.array(data, dtype=float)
    except ValueError:
        # Fallback for ragged arrays
        return np.array(data, dtype=object)


def get_ground_truth_from_data(raw_data: np.ndarray) -> Dict[int, float]:
    """
    Processes raw data (index, cost, ...) into a dictionary {k: min_cost}.
    Assumes index is the first column and cost is the second.
    """
    ground_truth = {}
    for row in raw_data:
        # Assuming format: (index, cost, ...)
        # We cast index back to int for bit counting
        idx = int(row[0])
        cost = row[1]
        
        k = bin(idx).count('1')
        # We want the minimum cost for each k
        if k not in ground_truth or cost < ground_truth[k]:
            ground_truth[k] = cost
    return ground_truth 

def load_bell_expression(name: str) -> Dict[str, float]:
    """
    Loads a Bell expression from a JSON file.
    
    Parameters
    ----------
    name : str
        The name of the Bell expression (e.g., "I_3322_std").
        
    dict
        The Bell expression dictionary.
    """
    # Assuming data is in ../../../data/raw relative to this file
    # src/optimalsdp/bell_utils.py -> src/optimalsdp -> src -> root -> data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_dir = os.path.join(base_dir, "data", "raw")
    
    # Possible subdirectories to search
    subdirs = ["3322", "4422"]
    
    filepath = None
    for subdir in subdirs:
        potential_path = os.path.join(raw_data_dir, subdir, f"{name}.json")
        if os.path.exists(potential_path):
            filepath = potential_path
            break
            
    if filepath is None:
        # Fallback: check if it's directly in raw_data_dir or if user provided a path
        potential_path = os.path.join(raw_data_dir, f"{name}.json")
        if os.path.exists(potential_path):
            filepath = potential_path
        else:
             raise FileNotFoundError(f"Bell expression file not found for {name} in {raw_data_dir} subdirectories {subdirs}")
        
    with open(filepath, 'r') as f:
        bell_expression = json.load(f)
        
    return bell_expression

#####CLASSES#####
#BELL INEQUALITIES
class Behaviour():
    
    def __init__(self, m_a: int, m_b: int, d_a: int, d_b: int) -> None:
        self.m_a = m_a
        self.m_b = m_b
        self.d_a = d_a
        self.d_b = d_b
        self.A_proj_meas, self.B_proj_meas, self.proj_meas = self.generate_proj_measurements()
        
    def generate_proj_measurements(self):
        result = [""]
        A_meas = []
        if self.d_a == 2:
            for i in range(1, self.m_a + 1):
                A_meas.append(f"A{i}")
                result.append(f"A{i}")
        else:
            for i in range(1, self.m_a + 1):
                for j in range(1, self.d_a):
                    A_meas.append(f"A{i}{j}")
                    result.append(f"A{i}{j}")

        B_meas = []
        if self.d_b == 2:
            for i in range(1, self.m_b + 1):
                B_meas.append(f"B{i}")
                result.append(f"B{i}")
        else:
            for i in range(1, self.m_b + 1):
                for j in range(1, self.d_b):
                    B_meas.append(f"B{i}{j}")
                    result.append(f"B{i}{j}")
        
        return A_meas, B_meas, result
        

    def generate_npa(self, npa_string):
        if npa_string.startswith("local"):
            local1 = self.proj_meas.copy()
            for a in self.A_proj_meas:
                for b in self.B_proj_meas:
                    local1.append(a + b)
            return local1

        elif not npa_string.startswith("npa"):
            raise ValueError("Input string must start with 'npa'")
        
        else:
            try:
                npa_level = int(npa_string[3:])
            except ValueError:
                raise ValueError("The part after 'npa' must be an integer.")
        
        not_simplified_yet = self._npa_recursive(npa_level)
        return not_simplified_yet
        
    
    def _npa_recursive(self, level):
        if level == 1:
            return self.proj_meas
        else:
            previous_level = self._npa_recursive(level - 1)
            result = []
            for a in self.proj_meas:
                for b in previous_level:
                    simplified_op = simplify_algebra_bell(a, b)
                    if simplified_op not in result and simplified_op!= "0":
                        result.append(simplified_op)
            return result
    
class MomentMatrix_Bell():
    def __init__(self, npa_proj_list: List[str]) -> None:
        self.matrix, self.projectors_indices = self._generate_moment_matrix(npa_proj_list)
        
    def _generate_moment_matrix(self, proj_list):
        matrix = []
        operators_indices = {}
        
        for i in range(len(proj_list)):
            matrix_row = []
            for j in range(len(proj_list)):
                simplified_op = simplify_algebra_bell(conjugate_operator_bell(proj_list[i]), proj_list[j])
                matrix_row.append(simplified_op)
                if simplified_op in operators_indices:
                    if (int(i), int(j)) not in operators_indices[simplified_op]:
                        operators_indices[simplified_op].append((int(i), int(j)))
                else:
                    operators_indices[simplified_op] = [(int(i), int(j))]
            matrix.append(matrix_row)
        return np.array(matrix), operators_indices
     
    
    def inequality_violation(self, bell_ineq: Dict[str, float]) -> Tuple[Optional[np.ndarray], float]:
        
        C = np.zeros((len(self.matrix), len(self.matrix)), dtype=float)
        for op, coeff in bell_ineq.items():
            if op not in self.projectors_indices:
                continue
            indices_list = self.projectors_indices[op]
            normalization_factor = 1.0/len(indices_list)
            for indices in indices_list:
                i, j = int(indices[0]), int(indices[1])
                C[i, j] = float(coeff * normalization_factor)
        
        Gamma = cp.Variable((len(C), len(C)))
        constraints = [Gamma>>0, Gamma[0,0]==1]
        op_seen = set()
        for op, idx_list in self.projectors_indices.items():
            if op not in op_seen:
                ref_i, ref_j = idx_list[0]
                ref_i, ref_j = int(ref_i), int(ref_j)
                for (i, j) in idx_list[1:]:  
                    i, j = int(i), int(j)
                    constraints.append(Gamma[i, j] == Gamma[ref_i, ref_j]) 
                if conjugate_operator_bell(op) in self.projectors_indices and conjugate_operator_bell(op)!= op:
                    for (k, l) in self.projectors_indices[conjugate_operator_bell(op)]:
                        k, l = int(k), int(l)
                        constraints.append(Gamma[k, l] == cp.conj(Gamma[ref_i, ref_j]))
                    op_seen.add(conjugate_operator_bell(op))
                    
                op_seen.add(op) 
                    
            else:
                continue
        C_const = cp.Constant(C)
        #problem_diagnose(C, Gamma, constraints, self.projectors_indices, op_exprs=bell_ineq)
        objective = cp.Maximize(cp.trace(C_const.T @ Gamma))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver="MOSEK")

        
        return Gamma.value, problem.value

