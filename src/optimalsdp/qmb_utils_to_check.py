#QMB
def conjugate_operator_qmb(monom1):
    """
    Parameters
    ----------
    monom1 : tupl
        Takes a product of operators described by a string.

    Returns
    -------
    monom1_conjugate : tupl
        Returns the conjugate of the initial product of operators.
    """
    monom1_s = monom1[1]
    monom1_list = []
    while monom1_s:
        i = 1
        while not monom1_s[i].isalpha():
            i += 1
        monom1_list.append(monom1_s[:i+1])
        monom1_s = monom1_s[i+1:]
    
    s1_dagg = ""
    for i in range(1, len(monom1_list)+1):
        s1_dagg += monom1_list[-i]
        
    monom1_conjugate = (np.conjugate(monom1[0]), s1_dagg)
    return monom1_conjugate

def simplify_algebra_qmb(monom1, monom2):
    """
    PARAMETERS
    ----------
    monom1 : tuple(int coefficient, str operator)
    monom2 : tuple(int coefficient, str operator)
        Each operator is described by a tuple which contains 
        the coefficient of the operator and te operator itself.
        For instance: 6*A := (6, "A")

    RETURNS
    -------
    new_tuple : tuple(int coefficient, str operator)
        Returns a simplified tuple of the simplified tuple
        For instance: 2 * σˣ₁ * σʸ₁ --> (2i, "σz1")
    """

    coeff = monom1[0] * monom2[0]
    s = monom1[1] + monom2[1]
         
    atoms = {}
    while s:
        i = 2
        while not s[i].isalpha():
            i += 1
        atom = s[1:i]
        if "\u03C3"+atom in atoms:
            atoms["\u03C3"+atom] += s[i]
        else:
            atoms["\u03C3"+atom] = s[i]   
        s = s[i+1:]       
    atoms = dict(sorted(atoms.items(), key=lambda x: int(x[0][1:])))
    
    commutation_rules = {"xx": (1, ""),
                         "yy": (1, ""),
                         "zz": (1, ""),
                         "xy": (1j, "z"), "yx": (-1j, "z"),
                         "yz": (1j, "x"), "zy": (-1j, "x"),
                         "zx": (1j, "y"), "xz": (-1j, "y")}
    
    atoms_simplified = {}
    for atom, op in atoms.items():
        while op[:2] in commutation_rules:
            coeff *= commutation_rules[op[:2]][0]
            op = commutation_rules[op[:2]][1] + op[2:]
        atoms_simplified[atom] = op
    
    final_op = ""
    for atom, op in atoms_simplified.items():
        if op:
            final_op += atom+op
    
    if coeff.imag == 0:
        coeff = int(coeff.real)
    
    return (coeff, final_op)

def print_as_strings(op_list):
    new_op_list = []
    for op in op_list:
        if op[0] == 1:
            new_op_list.append(op[1])
        elif op[0] == -1:
            new_op_list.append("-" + op[1])
        elif  op[0] == 1j:
            new_op_list.append("i*" + op[1])
        elif  op[0] == -1j:
            new_op_list.append("-i*" + op[1])
    print(new_op_list)
    
def metropolis_hastings_mc_QMB(starting_set, relaxation_set, hamiltonian, n_selected=20, steps=100, T_starting =1.0, alpha = None, initial_guess = None):
    """
    Implements the Metropolis-Hastings Monte Carlo algorithm for selecting moments.
    
    Parameters:
        n_moments (int): Total number of possible moments.
        n_selected (int): Number of moments to be selected.
        steps (int): Number of Monte Carlo steps.
        temperature (float): Initial temperature for simulated annealing (if needed).
    
    Returns:
        best_selection (np.ndarray): Best selection found.
        best_cost (float): Cost corresponding to the best selection.
    """
    
    n_moments = len(relaxation_set)
    selection = np.zeros(n_moments, dtype=int)
    
    # We define a one hot-encoding array for the selected monomials 
    # to be added
    if initial_guess is not None:
        initial_guess = np.array(initial_guess)
        if len(initial_guess) != len(relaxation_set):
            ValueError("initial_guess and relaxation_set should have same length!")
        selection[initial_guess == 1] = 1    
        available_indices = np.where(initial_guess == 0)[0]
        random_indices = np.random.choice(available_indices, n_selected, replace=False)
        selection[random_indices] = 1
    else:
        selection[np.random.choice(n_moments, n_selected, replace=False)] = 1
    
    # Calculate the value for the moment matrix built from the lower level
    # plus the selcted monomials from the upper level
    
    add_from_relaxation_set = [relaxation_set[i] for i in range(n_moments) if selection[i] == 1]
    moment_matrix = MomentMatrix_Bell(starting_set + add_from_relaxation_set)
    _, current_cost = moment_matrix.find_lower_bound(hamiltonian)
    
    best_selection = selection.copy()
    best_cost = current_cost
    
    all_selections = []
    all_costs = []
    for step in range(steps):
        if alpha is not None:
            temperature = T_starting *(alpha**step)
        else:
            temperature = T_starting
        
        # Select a random swap 
        ones_indices = np.where(selection == 1)[0]
        zeros_indices = np.where(selection == 0)[0]
        swap_in = random.choice(zeros_indices)
        swap_out = random.choice(ones_indices)
        selection[swap_out], selection[swap_in] = selection[swap_in], selection[swap_out]
        
        # Calculate new value (new cost)
        new_add_from_relaxation_set = [relaxation_set[i] for i in range(n_moments) if selection[i] == 1]
        new_moment_matrix = MomentMatrix_QMB(starting_set + new_add_from_relaxation_set)
        _, new_cost = new_moment_matrix.find_lower_bound(hamiltonian)
        
        # Accept or reject the move
        delta_cost = new_cost - current_cost
        if delta_cost > 0 or np.exp(delta_cost / temperature) > np.random.rand():
            # if delta_cost > 0:
            #     print(f"Acceptance probability of this move was: {np.exp(-delta_cost / temperature)}")
            current_cost = new_cost
            all_selections.append(selection)
            all_costs.append(current_cost)
            if new_cost > best_cost:
                best_cost = new_cost
                best_selection = selection.copy()         
        else:
            # Revert swap if move is rejected
            selection[swap_out], selection[swap_in] = selection[swap_in], selection[swap_out]
            all_selections.append(selection)
            all_costs.append(current_cost)
               
    return best_selection, best_cost, all_selections, all_costs



#QMB
class Hamiltonian():
    
    def __init__(self, N, J=1, h=.5, K=0, model="Ising"):

        self.atoms = N
        self.J = J
        self.h = h
        self.K = K
        self.model = model
        self.operators = self._generate_operators()
        self.expression = self._generate_expression()
        
    def _generate_operators(self):
        """
        Returns: List of the pauli matrices taken into account for a 
        specific Hamiltonian.
        -------
        monomials : list[tuple]
            A list of tuples where each tuple represent the coefficient of
            the operator and the operator itself.
        """
        monomials = [(1, "")]
        for i in range(1, self.atoms+1):
            for coord in ['x', 'y', 'z']:
                monomials.append((1, "\u03C3"+str(i)+coord))
        return monomials
    
    def _generate_expression(self):
        """
        Returns: The expression of the Hamiltonian
        -------
        expression : dict
            The Hamiltonian is defined as a dictionary containing the
            operators as keys and their respective coefficients as 
            their values. (Same conevntion for the Bell inequalities)
        """
        expression = {}
        if self.model == "Ising":
            for i in range(1, self.atoms):
                if self.J != 0:
                    expression["\u03C3"+str(i)+"z" + "\u03C3"+str(i+1)+"z"] = self.J
                if self.h != 0:
                    expression["\u03C3"+str(i)+"x"] = -1*self.h
                    
        if self.model == "Heisenberg":
            for i in range(1, self.atoms):
                for coord in ['x', 'y', 'z']:
                    expression["\u03C3"+str(i)+coord + "\u03C3"+str(i+1)+coord ] = self.J
            for coord in ['x', 'y', 'z']:
                expression["\u03C3"+str(1)+coord + "\u03C3"+str(self.atoms)+coord ] = self.J
        
        return expression
            
    def _npa_recursive(self, level):
        if level == 1:
            return self.operators
        else:
            previous_level = self._npa_recursive(level - 1)
            result = []
            for a in self.operators:
                for b in previous_level:
                    simplified_op = simplify_algebra_qmb(a, b)
                    if simplified_op not in result:
                        result.append(simplified_op)
            return sorted(result, key = lambda x: len(x[1]))
        
    def generate_npa(self, level_string):
        if not level_string.startswith("level"):
            raise ValueError("Input string must start with 'level'")
        else:
            npa_level = int(level_string[5:])
            return self._npa_recursive(npa_level)

class MomentMatrix_QMB():
    def __init__(self, npa_op_list):
        self.matrix, self.values, self.operators_indices = self._generate_moment_matrix(npa_op_list)
        
    def _generate_moment_matrix(self, op_list):
        matrix = np.zeros((len(op_list), len(op_list)), dtype=tuple)
        M_matrix = []
        M_values = []
        operators_indices = {}
        
        for i in range(len(op_list)):
            M_matrix_row = []
            M_values_row = []
            for j in range(len(op_list)):
                simplified_op = simplify_algebra_qmb(conjugate_operator_qmb(op_list[i]), op_list[j])
                matrix[i, j] = simplified_op
                M_matrix_row.append(simplified_op[1])
                M_values_row.append(simplified_op[0])
                if simplified_op in operators_indices:
                    operators_indices[simplified_op].append((i, j))
                else:
                    operators_indices[simplified_op] = [(i, j)]
            M_matrix.append(M_matrix_row)
            M_values.append(M_values_row)
        
        return np.array(M_matrix), np.array(M_values), operators_indices

        
    def find_lower_bound(self, hamiltonian):
        
        expression = hamiltonian.expression
        H_m = np.zeros((len(self.matrix), len(self.matrix)))
        for op, coeff in expression.items():
            if op not in self.projectors_indices:
                continue
            indices_list = self.operators_indices[(1, op)]
            normalization_factor = 1/(len(indices_list))
            for indices in indices_list:
                i, j = int(indices[0]), int(indices[1])
                H_m[i, j] = float(coeff*normalization_factor)
                                   
        Gamma = cp.Variable((len(H_m), len(H_m)))
        constraints = [Gamma>>0, cp.diag(Gamma)==1]
        op_seen = set()
        for op, idx_list in self.operators_indices.items():
            """
            Input known values of Gamma as constraints.
            Those values are [1, -1, i, -i].
            Impose as equalities all the entries of Gamma
            where the same operator appears.
            Impose the entries that are complex conjugates one from
            the other.
            """
            
            if op not in op_seen:
                ref_i, ref_j = idx_list[0]
                ref_i, ref_j = int(ref_i), int(ref_j)
                for (i, j) in idx_list[1:]:
                    i, j = int(i), int(j)
                    constraints.append(Gamma[i, j] == Gamma[ref_i, ref_j])
                
                conj_op = (np.conjugate(op[0]), op[1])
                if conj_op in self.operators_indices and conj_op!=op:
                    for (k, l) in self.operators_indices[conj_op]:
                        k, l = int(k), int(l)
                        constraints.append(Gamma[k, l] == cp.conj(Gamma[ref_i, ref_j]))
                    op_seen.add(conj_op)
                
                minus_op = (-op[0], op[1])
                if minus_op in self.operators_indices:  
                    for (k, l) in self.operators_indices[minus_op]:
                        k, l = int(k), int(l)
                        constraints.append(Gamma[k, l] == -Gamma[ref_i, ref_j])
                    op_seen.add(minus_op)
                    
                op_seen.add(op)
                    
            else:
                continue
                
        H_m = cp.Constant(H_m)     
        objective = cp.Minimize(cp.trace(H_m.T @ Gamma))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver = cp.SCS)
        
        return Gamma.value, problem.value/hamiltonian.atoms
        
