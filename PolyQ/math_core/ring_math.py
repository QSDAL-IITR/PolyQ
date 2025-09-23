"""
Unified ring mathematics and quadratic forms using SymPy for exact computation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Union, Optional, Set
import sympy as sp
from sympy import Matrix, symbols, Poly, GF, ZZ, Rational, Mod
from sympy.polys.galoistools import gf_irreducible_p
from sympy.polys.domains import ZZ as ZZ_domain
from functools import lru_cache
import numpy as np
import time

class EnhancedRing(ABC):
    """Abstract base class for mathematical rings using SymPy."""
    
    @abstractmethod
    def domain(self):
        """Return the SymPy domain for this ring."""
        pass
    
    @abstractmethod
    def zero(self):
        """Return the additive identity."""
        pass
    
    @abstractmethod
    def one(self):
        """Return the multiplicative identity."""
        pass
    
    @abstractmethod
    def characteristic(self):
        """Return the characteristic of the ring."""
        pass

class ZnRing(EnhancedRing):
    """Ring Z/nZ using SymPy modular arithmetic with optimization."""
    
    def __init__(self, modulus: int):
        self.modulus = modulus
        self._is_prime = sp.isprime(modulus)
        self._characteristic = modulus if self._is_prime else 0
        
        # Use GF for prime modulus
        if self._is_prime:
            self._domain = GF(modulus)
            self._zero = self._domain.zero
            self._one = self._domain.one
        else:
            self._domain = None
            self._zero = 0
            self._one = 1
    
    def domain(self):
        return self._domain if self._domain else f"Z/{self.modulus}Z"
    
    def zero(self):
        return self._zero
    
    def one(self):
        return self._one
    
    def characteristic(self):
        return self._characteristic
    
    def element(self, value: int):
        """Create ring element from integer."""
        if self._domain:
            return self._domain(value % self.modulus)
        return value % self.modulus
    
    def add(self, a, b):
        """Addition in Z/nZ."""
        if self._domain:
            return self._domain(a) + self._domain(b)
        return (int(a) + int(b)) % self.modulus
    
    def multiply(self, a, b):
        """Multiplication in Z/nZ."""
        if self._domain:
            return self._domain(a) * self._domain(b)
        return (int(a) * int(b)) % self.modulus
    
    @lru_cache(maxsize=256)
    def inverse(self, a):
        """Multiplicative inverse if it exists."""
        a_int = int(a) % self.modulus
        if a_int == 0:
            return None
            
        if self._domain:
            try:
                return self._domain(a_int) ** (-1)
            except ZeroDivisionError:
                return None
        else:
            return self._mod_inverse(a_int, self.modulus)
    
    def _mod_inverse(self, a: int, m: int) -> Optional[int]:
        """Compute modular inverse using extended Euclidean algorithm."""
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd_val, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd_val, x, y
        
        gcd_val, x, _ = extended_gcd(a % m, m)
        if gcd_val != 1:
            return None
        return (x % m + m) % m
    
    def is_unit(self, a) -> bool:
        """Check if element is invertible."""
        return self.inverse(a) is not None
    
    def __str__(self):
        return f"Z/{self.modulus}Z"

class FiniteField(EnhancedRing):
    """Finite field GF(p^n) using SymPy."""
    
    def __init__(self, characteristic: int, degree: int = 1):
        if not sp.isprime(characteristic):
            raise ValueError(f"Characteristic {characteristic} must be prime")
            
        self._characteristic = characteristic
        self.degree = degree
        self.order = characteristic ** degree
        
        if degree == 1:
            self._domain = GF(characteristic)
        else:
            self._domain = GF(characteristic**degree)
    
    def domain(self):
        return self._domain
    
    def zero(self):
        return self._domain.zero
    
    def one(self):
        return self._domain.one
    
    def characteristic(self):
        return self._characteristic
    
    def element(self, value):
        """Create field element."""
        return self._domain(value)
    
    def __str__(self):
        if self.degree == 1:
            return f"GF({self._characteristic})"
        return f"GF({self._characteristic}^{self.degree})"

class UnifiedQuadraticForm:
    """Unified quadratic forms supporting both SymPy and numpy backends."""
    
    def __init__(self, coefficients: Dict[Tuple[int, int], Any], ring: EnhancedRing):
        self.ring = ring
        self.raw_coeffs = coefficients.copy()
        
        # Normalize and store coefficients
        self.coeffs = {}
        for (i, j), v in coefficients.items():
            if hasattr(ring, 'element'):
                self.coeffs[(i, j)] = ring.element(v)
            else:
                self.coeffs[(i, j)] = v
        
        self.n_vars = max(max(k) for k in coefficients.keys()) + 1 if coefficients else 0
        
        # Cache matrices
        self._symmetric_matrix = None
        self._bilinear_matrix = None
        self._numpy_matrix = None
    
    def to_symmetric_matrix(self) -> Matrix:
        """Convert to symmetric matrix (cached)."""
        if self._symmetric_matrix is not None:
            return self._symmetric_matrix
            
        matrix_data = []
        for i in range(self.n_vars):
            row = []
            for j in range(self.n_vars):
                if (i, j) in self.coeffs:
                    row.append(self.coeffs[(i, j)])
                elif (j, i) in self.coeffs:
                    row.append(self.coeffs[(j, i)])
                else:
                    row.append(self.ring.zero())
            matrix_data.append(row)
        
        self._symmetric_matrix = Matrix(matrix_data)
        return self._symmetric_matrix
    
    def to_bilinear_matrix(self) -> Matrix:
        """Convert to bilinear form matrix (cached)."""
        if self._bilinear_matrix is not None:
            return self._bilinear_matrix
        
        matrix_data = []
        char = getattr(self.ring, 'characteristic', lambda: 0)()
        
        for i in range(self.n_vars):
            row = []
            for j in range(self.n_vars):
                if i == j and (i, j) in self.coeffs:
                    # Diagonal elements
                    row.append(self.coeffs[(i, j)])
                elif i != j:
                    # Off-diagonal elements
                    coeff = None
                    if (i, j) in self.coeffs:
                        coeff = self.coeffs[(i, j)]
                    elif (j, i) in self.coeffs:
                        coeff = self.coeffs[(j, i)]
                    
                    if coeff is not None:
                        if char == 2:
                            # In characteristic 2, no division by 2 needed
                            row.append(coeff)
                        else:
                            # Try to divide by 2
                            try:
                                two_inv = self.ring.inverse(2)
                                if two_inv is not None:
                                    row.append(self.ring.multiply(coeff, two_inv))
                                else:
                                    row.append(coeff)
                            except:
                                row.append(coeff)
                    else:
                        row.append(self.ring.zero())
                else:
                    row.append(self.ring.zero())
            matrix_data.append(row)
        
        self._bilinear_matrix = Matrix(matrix_data)
        return self._bilinear_matrix
    
    def to_numpy_matrix(self) -> np.ndarray:
        """Convert to numpy array for fast computation."""
        if self._numpy_matrix is not None:
            return self._numpy_matrix
        
        matrix = np.zeros((self.n_vars, self.n_vars), dtype=int)
        
        # Get modulus if available
        modulus = getattr(self.ring, 'modulus', None)
        
        for (i, j), coeff in self.coeffs.items():
            coeff_int = int(coeff) if hasattr(coeff, '__int__') else coeff
            if modulus:
                coeff_int = coeff_int % modulus
            
            if i == j:
                matrix[i, j] = coeff_int
            else:
                # Handle bilinear form conversion
                if modulus == 2:
                    matrix[i, j] = coeff_int
                    matrix[j, i] = coeff_int
                else:
                    # Try to divide by 2
                    if modulus and modulus % 2 == 1:
                        inv_2 = self._fast_mod_inverse(2, modulus)
                        if inv_2:
                            half_coeff = (coeff_int * inv_2) % modulus
                            matrix[i, j] = half_coeff
                            matrix[j, i] = half_coeff
                        else:
                            matrix[i, j] = coeff_int
                            matrix[j, i] = coeff_int
                    else:
                        matrix[i, j] = coeff_int
                        matrix[j, i] = coeff_int
        
        if modulus:
            matrix = matrix % modulus
        
        self._numpy_matrix = matrix
        return self._numpy_matrix
    
    @lru_cache(maxsize=128)
    def _fast_mod_inverse(self, a: int, m: int) -> Optional[int]:
        """Fast modular inverse computation."""
        def gcd_extended(a, b):
            if a == 0:
                return b, 0, 1
            gcd_val, x1, y1 = gcd_extended(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd_val, x, y
        
        gcd_val, x, _ = gcd_extended(a % m, m)
        if gcd_val != 1:
            return None
        return (x % m + m) % m

class HybridLinearAlgebra:
    """Hybrid linear algebra using both SymPy and optimized numpy computation."""
    
    def __init__(self, ring: EnhancedRing):
        self.ring = ring
        self.modulus = getattr(ring, 'modulus', None)
        self.is_finite = self.modulus is not None or hasattr(ring, 'order')
        self.use_fast_path = self.modulus and self.modulus < 100
    
    def eigenvals_eigenvects(self, quad_form: UnifiedQuadraticForm) -> List[Tuple[Any, int, List[Matrix]]]:
        """Compute eigenvalues and eigenvectors using optimal method."""
        
        # Choose computation method based on ring properties
        if self.use_fast_path:
            return self._fast_eigenvals(quad_form)
        elif self.is_finite:
            return self._sympy_finite_eigenvals(quad_form.to_bilinear_matrix())
        else:
            return self._sympy_general_eigenvals(quad_form.to_bilinear_matrix())
    
    def _fast_eigenvals(self, quad_form: UnifiedQuadraticForm) -> List[Tuple[Any, int, List[Matrix]]]:
        """Fast eigenvalue computation using numpy."""
        matrix = quad_form.to_numpy_matrix()
        n = matrix.shape[0]
        eigenvals = []
        
        for val in range(self.modulus):
            char_matrix = (matrix - val * np.eye(n, dtype=int)) % self.modulus
            
            # Fast singularity check
            if self._is_singular_fast(char_matrix):
                # Find eigenvectors
                eigenvects = self._null_space_fast(char_matrix)
                if eigenvects:
                    sympy_vects = [Matrix(v) for v in eigenvects]
                    eigenval = self.ring.element(val) if hasattr(self.ring, 'element') else val
                    eigenvals.append((eigenval, len(sympy_vects), sympy_vects))
        
        return eigenvals
    
    def _is_singular_fast(self, matrix: np.ndarray) -> bool:
        """Fast singularity check using rank computation."""
        return self._matrix_rank_fast(matrix) < matrix.shape[0]
    
    def _matrix_rank_fast(self, matrix: np.ndarray) -> int:
        """Fast matrix rank computation over Z/mZ."""
        A = matrix.copy()
        m, n = A.shape
        rank = 0
        
        for col in range(min(m, n)):
            # Find pivot
            pivot_row = None
            for row in range(rank, m):
                if A[row, col] % self.modulus != 0:
                    if self._gcd(A[row, col], self.modulus) == 1:
                        pivot_row = row
                        break
            
            if pivot_row is None:
                continue
            
            # Swap rows
            if pivot_row != rank:
                A[[rank, pivot_row]] = A[[pivot_row, rank]]
            
            # Get inverse
            pivot_inv = self._fast_mod_inverse(A[rank, col], self.modulus)
            if pivot_inv is None:
                continue
            
            # Eliminate
            for row in range(m):
                if row != rank and A[row, col] % self.modulus != 0:
                    mult = (A[row, col] * pivot_inv) % self.modulus
                    A[row] = (A[row] - mult * A[rank]) % self.modulus
            
            rank += 1
        
        return rank
    
    def _null_space_fast(self, matrix: np.ndarray) -> List[List[int]]:
        """Fast null space computation."""
        A = matrix.copy()
        m, n = A.shape
        
        # Gaussian elimination
        pivot_cols = []
        rank = 0
        
        for col in range(n):
            pivot_row = None
            for row in range(rank, m):
                if A[row, col] % self.modulus != 0 and self._gcd(A[row, col], self.modulus) == 1:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                continue
            
            if pivot_row != rank:
                A[[rank, pivot_row]] = A[[pivot_row, rank]]
            
            pivot_cols.append(col)
            pivot_inv = self._fast_mod_inverse(A[rank, col], self.modulus)
            
            if pivot_inv is not None:
                A[rank] = (A[rank] * pivot_inv) % self.modulus
                
                for row in range(m):
                    if row != rank and A[row, col] % self.modulus != 0:
                        mult = A[row, col]
                        A[row] = (A[row] - mult * A[rank]) % self.modulus
            
            rank += 1
        
        # Find free variables
        free_vars = [i for i in range(n) if i not in pivot_cols]
        if not free_vars:
            return []
        
        # Generate basis vectors
        null_vectors = []
        for free_var in free_vars:
            vec = [0] * n
            vec[free_var] = 1
            
            # Back substitution
            for i in range(rank-1, -1, -1):
                if i < len(pivot_cols):
                    col = pivot_cols[i]
                    val = 0
                    for j in range(col+1, n):
                        val = (val - A[i, j] * vec[j]) % self.modulus
                    vec[col] = val % self.modulus
            
            null_vectors.append(vec)
        
        return null_vectors
    
    def _sympy_finite_eigenvals(self, matrix: Matrix) -> List[Tuple[Any, int, List[Matrix]]]:
        """SymPy eigenvalue computation for finite rings."""
        n = matrix.rows
        eigenvals = []
        
        # Determine test range
        if hasattr(self.ring, 'modulus'):
            test_range = range(self.ring.modulus)
        elif hasattr(self.ring, 'order'):
            test_range = range(min(self.ring.order, 50))
        else:
            test_range = range(20)
        
        for val in test_range:
            try:
                eigenval = self.ring.element(val) if hasattr(self.ring, 'element') else val
                char_matrix = matrix - eigenval * sp.eye(n)
                
                # Check if singular
                try:
                    det = char_matrix.det()
                    is_zero = (det == self.ring.zero() if hasattr(self.ring, 'zero') else det == 0)
                    
                    if is_zero:
                        eigenvects = char_matrix.nullspace()
                        if eigenvects:
                            eigenvals.append((eigenval, len(eigenvects), eigenvects))
                            
                except Exception:
                    continue
                    
            except Exception:
                continue
        
        return eigenvals
    
    def _sympy_general_eigenvals(self, matrix: Matrix) -> List[Tuple[Any, int, List[Matrix]]]:
        """General SymPy eigenvalue computation."""
        try:
            return matrix.eigenvects()
        except Exception:
            return []
    
    @lru_cache(maxsize=1024)
    def _gcd(self, a: int, b: int) -> int:
        """Cached GCD computation."""
        while b:
            a, b = b, a % b
        return abs(a)
    
    @lru_cache(maxsize=512)
    def _fast_mod_inverse(self, a: int, m: int) -> Optional[int]:
        """Fast cached modular inverse."""
        a = a % m
        if self._gcd(a, m) != 1:
            return None
        
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd_val, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd_val, x, y
        
        _, x, _ = extended_gcd(a, m)
        return (x % m + m) % m
    
    def diagonalize_symmetric(self, quad_form: UnifiedQuadraticForm) -> Tuple[Optional[Matrix], Optional[Matrix], bool]:
        """Diagonalize symmetric matrix using congruence transformation."""
        if self.use_fast_path:
            return self._fast_diagonalize_symmetric(quad_form)
        else:
            return self._sympy_diagonalize_symmetric(quad_form.to_bilinear_matrix())
    
    def _fast_diagonalize_symmetric(self, quad_form: UnifiedQuadraticForm) -> Tuple[Optional[Matrix], Optional[Matrix], bool]:
        """Fast symmetric diagonalization using numpy."""
        matrix = quad_form.to_numpy_matrix()
        n = matrix.shape[0]
        A = matrix.copy()
        P = np.eye(n, dtype=int)
        
        for i in range(n):
            # Find pivot
            pivot_found = False
            
            if A[i, i] != 0 and self._gcd(A[i, i], self.modulus) == 1:
                pivot_found = True
            else:
                # Try to create a pivot
                for j in range(i, n):
                    for k in range(j, n):
                        if A[j, k] != 0 and self._gcd(A[j, k], self.modulus) == 1:
                            if j != i:
                                A[[i, j]] = A[[j, i]]
                                A[:, [i, j]] = A[:, [j, i]]
                                P[[i, j]] = P[[j, i]]
                            
                            if k != i and k != j:
                                for row in range(n):
                                    A[row, i] = (A[row, i] + A[row, k]) % self.modulus
                                for col in range(n):
                                    A[i, col] = (A[i, col] + A[k, col]) % self.modulus
                                for row in range(n):
                                    P[row, i] = (P[row, i] + P[row, k]) % self.modulus
                            
                            pivot_found = True
                            break
                    if pivot_found:
                        break
            
            if not pivot_found:
                return None, None, False
            
            # Eliminate
            pivot = A[i, i]
            pivot_inv = self._fast_mod_inverse(pivot, self.modulus)
            
            if pivot_inv is None:
                return None, None, False
            
            for j in range(i+1, n):
                if A[i, j] != 0:
                    multiplier = (A[i, j] * pivot_inv) % self.modulus
                    
                    for k in range(n):
                        A[k, j] = (A[k, j] - multiplier * A[k, i]) % self.modulus
                        A[j, k] = (A[j, k] - multiplier * A[i, k]) % self.modulus
                        P[k, j] = (P[k, j] - multiplier * P[k, i]) % self.modulus
        
        return Matrix(P), Matrix(A), True
    
    def _sympy_diagonalize_symmetric(self, matrix: Matrix) -> Tuple[Optional[Matrix], Optional[Matrix], bool]:
        """SymPy symmetric diagonalization."""
        try:
            # For small matrices, try direct diagonalization
            if matrix.rows <= 3:
                P, D = matrix.diagonalize()
                return P, D, True
            else:
                # For larger matrices, this is more complex in general rings
                return None, None, False
        except Exception:
            return None, None, False

def comprehensive_example():
    """Comprehensive example showcasing all features."""
    print("=" * 60)
    print("COMPREHENSIVE QUADRATIC FORM ANALYSIS")
    print("=" * 60)
    
    examples = [
        (ZnRing(5), "Z/5Z", {(0, 0): 2, (0, 1): 3, (1, 1): 1}),
        (ZnRing(7), "Z/7Z", {(0, 0): 1, (0, 1): 2, (1, 1): 3}),
        (FiniteField(3), "GF(3)", {(0, 0): 1, (0, 1): 2, (1, 1): 2}),
    ]
    
    for ring, name, coeffs in examples:
        print(f"\n{name} Example:")
        print("-" * 40)
        
        start_time = time.time()
        
        # Create quadratic form
        quad_form = UnifiedQuadraticForm(coeffs, ring)
        algebra = HybridLinearAlgebra(ring)
        
        # Display matrices
        print("Bilinear matrix:")
        sp.pprint(quad_form.to_bilinear_matrix())
        
        # Compute eigenvalues
        eigenvals = algebra.eigenvals_eigenvects(quad_form)
        
        print(f"\nEigenvalues and eigenvectors:")
        for eigenval, mult, eigenvects in eigenvals:
            print(f"λ = {eigenval} (multiplicity: {mult})")
            for i, vec in enumerate(eigenvects):
                print(f"  v_{i+1} = {vec.T}")
        
        # Try diagonalization
        P, D, success = algebra.diagonalize_symmetric(quad_form)
        
        if success:
            print(f"\nSymmetric diagonalization successful:")
            print("P =")
            sp.pprint(P)
            print("D =")
            sp.pprint(D)
        else:
            print("\nSymmetric diagonalization failed")
        
        elapsed = time.time() - start_time
        print(f"\nComputation time: {elapsed:.4f}s")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    comprehensive_example()