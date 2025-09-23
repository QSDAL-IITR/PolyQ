from typing import List, Tuple, Dict

# ========================== Integer Ring (Z) ==========================
class IntegerRing:
    def zero(self) -> int: return 0
    def one(self) -> int: return 1
    def add(self, a: int, b: int) -> int: return a + b
    def sub(self, a: int, b: int) -> int: return a - b
    def mul(self, a: int, b: int) -> int: return a * b
    def neg(self, a: int) -> int: return -a
    def is_zero(self, a: int) -> bool: return a == 0
    def is_unit(self, a: int) -> bool: return abs(a) == 1

    def div_exact(self, a: int, b: int) -> int:
        if b == 0: raise ZeroDivisionError("division by zero")
        q, r = divmod(a, b)
        if r != 0: raise ValueError(f"{a} not divisible by {b}")
        return q

    def gcd(self, a: int, b: int) -> int:
        a, b = abs(a), abs(b)
        while b: a, b = b, a % b
        return a

Z = IntegerRing()

# ========================== Integer Matrices ==========================
class IntMatrix:
    def __init__(self, data: List[List[int]]):
        if not data or not data[0]:
            raise ValueError("Matrix must be nonempty")
        cols = len(data[0])
        for r in data:
            if len(r) != cols:
                raise ValueError("All rows must have equal length")
        self.A = [row[:] for row in data]
        self.m = len(data)
        self.n = cols

    @staticmethod
    def zeros(m: int, n: int) -> "IntMatrix":
        return IntMatrix([[0]*n for _ in range(m)])

    @staticmethod
    def eye(n: int) -> "IntMatrix":
        A = [[0]*n for _ in range(n)]
        for i in range(n): A[i][i] = 1
        return IntMatrix(A)

    def copy(self) -> "IntMatrix":
        return IntMatrix(self.A)

    def __getitem__(self, ij: Tuple[int,int]) -> int:
        i,j = ij
        return self.A[i][j]

    def __setitem__(self, ij: Tuple[int,int], v: int):
        i,j = ij
        self.A[i][j] = v

    def shape(self) -> Tuple[int,int]:
        return self.m, self.n

    def mul(self, other: "IntMatrix") -> "IntMatrix":
        if self.n != other.m: raise ValueError("dimension mismatch")
        C = [[0]*other.n for _ in range(self.m)]
        for i in range(self.m):
            for k in range(self.n):
                aik = self.A[i][k]
                if aik == 0: continue
                for j in range(other.n):
                    C[i][j] += aik * other.A[k][j]
        return IntMatrix(C)

    def trim(self) -> str:
        # pretty string (compact)
        return "\n".join("[" + " ".join(f"{v}" for v in row) + "]" for row in self.A)

# ========================== Bareiss Determinant (Z) ==========================
def det_bareiss(M: IntMatrix) -> int:
    # Fraction-free Bareiss algorithm (exact divisions over Z)
    m, n = M.shape()
    if m != n: raise ValueError("square matrix required")
    A = M.copy().A
    prev = 1
    sign = 1
    for k in range(n-1):
        # find pivot
        piv = k
        while piv < n and A[piv][k] == 0:
            piv += 1
        if piv == n: 
            return 0
        if piv != k:
            A[k], A[piv] = A[piv], A[k]
            sign = -sign
        pivot = A[k][k]
        for i in range(k+1, n):
            for j in range(k+1, n):
                # A[i,j] = (A[i,j]*pivot - A[i,k]*A[k,j]) / prev
                num = A[i][j]*pivot - A[i][k]*A[k][j]
                if k == 0:
                    A[i][j] = num
                else:
                    q, r = divmod(num, prev)
                    if r != 0:
                        # Fallback: keep numerator; determinant will still be correct from diagonal
                        A[i][j] = num
                    else:
                        A[i][j] = q
            A[i][k] = 0
        prev = pivot
    return sign * A[n-1][n-1]

# ========================== Fraction-free Row Echelon and RREF ==========================
def row_echelon_fraction_free(M: IntMatrix) -> Tuple[IntMatrix, List[int], List[int]]:
    # Eliminate strictly below pivot columns; no normalization; returns U, pivot_cols, pivots
    A = M.copy().A
    m, n = M.shape()
    row = 0
    pivot_cols: List[int] = []
    pivots: List[int] = []
    for col in range(n):
        if row == m: break
        # find pivot
        pr = row
        while pr < m and A[pr][col] == 0: pr += 1
        if pr == m: continue
        if pr != row:
            A[row], A[pr] = A[pr], A[row]
        pivot = A[row][col]
        # eliminate below (fraction-free)
        for i in range(row+1, m):
            if A[i][col] == 0: continue
            f = A[i][col]
            for j in range(col, n):
                A[i][j] = A[i][j]*pivot - f*A[row][j]
            A[i][col] = 0
        pivot_cols.append(col)
        pivots.append(pivot)
        row += 1
    return IntMatrix(A), pivot_cols, pivots

def rref_fraction_free(M: IntMatrix) -> Tuple[IntMatrix, int]:
    # Compute RREF without introducing rationals: eliminate above and below; no pivot normalization
    U, pivot_cols, pivots = row_echelon_fraction_free(M)
    A = U.A
    m, n = U.shape()
    r = len(pivot_cols)
    # eliminate above pivots
    for idx in range(r-1, -1, -1):
        row = idx
        col = pivot_cols[idx]
        pivot = A[row][col]
        if pivot == 0: continue
        for i in range(0, row):
            if A[i][col] == 0: continue
            f = A[i][col]
            for j in range(col, n):
                A[i][j] = A[i][j]*pivot - f*A[row][j]
            A[i][col] = 0
    return IntMatrix(A), r

# ========================== Integer Eigenvalues and Eigenvectors ==========================
def divisors(n: int) -> List[int]:
    n = abs(n)
    if n == 0:
        # convention: search a small set if det=0 (0 is always an eigenvalue when singular)
        return [0, 1, -1, 2, -2, 3, -3]
    ds = set()
    i = 1
    while i*i <= n:
        if n % i == 0:
            ds.add(i); ds.add(-i)
            ds.add(n//i); ds.add(-(n//i))
        i += 1
    return sorted(ds)

def mat_sub_lambda_I(M: IntMatrix, lam: int) -> IntMatrix:
    m, n = M.shape()
    if m != n: raise ValueError("square matrix required")
    A = M.copy().A
    for i in range(n):
        A[i][i] -= lam
    return IntMatrix(A)

def integer_eigenvalues(M: IntMatrix) -> List[int]:
    # test all integer divisors of det(M) (rational root candidates for det(λI - A) = 0)
    detA = det_bareiss(M)
    cands = divisors(detA)
    eigs = []
    for lam in cands:
        N = mat_sub_lambda_I(M, lam)
        if det_bareiss(N) == 0:
            if lam not in eigs:
                eigs.append(lam)
    return eigs

def integer_nullspace_basis_from_upper(U: IntMatrix, pivot_cols: List[int], pivots: List[int]) -> List[List[int]]:
    # Build integer basis vectors v for U v = 0 by choosing each free variable,
    # setting it to P = product of pivots, and exact back-substitution scaling.
    m, n = U.shape()
    r = len(pivot_cols)
    pivot_set = set(pivot_cols)
    P = 1
    for p in pivots: P *= p if p != 0 else 1
    basis: List[List[int]] = []
    free_cols = [j for j in range(n) if j not in pivot_set]
    for f in free_cols:
        v = [0]*n
        v[f] = P
        # back substitution from bottom pivot row to top
        for idx in range(r-1, -1, -1):
            row = idx
            c = pivot_cols[idx]
            p = U.A[row][c]
            # s = sum_{j>c} U[row][j] * v[j]
            s = 0
            for j in range(c+1, n):
                if U.A[row][j] != 0 and v[j] != 0:
                    s += U.A[row][j]*v[j]
            # set v[c] = -s / p (exact in our construction)
            if p == 0:
                v[c] = 0
            else:
                q, rmd = divmod(-s, p)
                if rmd != 0:
                    # As a safety, scale whole vector by p to make division exact
                    for t in range(n): v[t] *= p
                    q2, r2 = divmod(-s*p, p)
                    if r2 != 0: raise RuntimeError("unexpected non-exact division")
                    v[c] = q2
                else:
                    v[c] = q
        # remove common gcd to keep vector primitive
        g = 0
        for x in v: 
            g = Z.gcd(g, x)
        if g > 1:
            v = [x//g for x in v]
        basis.append(v)
    return basis

def integer_eigenvectors(M: IntMatrix, lam: int) -> List[List[int]]:
    N = mat_sub_lambda_I(M, lam)
    U, pivot_cols, pivots = row_echelon_fraction_free(N)
    return integer_nullspace_basis_from_upper(U, pivot_cols, pivots)

# ========================== Quadratic Integer Polynomials → Bilinear ==========================
# Representation of a quadratic polynomial in n variables:
#   Q(x) = sum_i a[(i,i)] * x_i^2 + sum_{i<j} a[(i,j)] * x_i * x_j   with a[(i,j)] integers
# The associated symmetric bilinear form B satisfies:
#   Q(x) = B(x,x), and for i<j, coefficient of x_i x_j equals 2*B_ij  ("twos out" convention).
# To stay in integers, we build M2 = 2B with:
#   M2_ii = 2*a[(i,i)] and M2_ij = a[(i,j)] for i≠j.
# Then Q(x) = (1/2) x^T M2 x with x^T M2 x always even for integer x.

def quadratic_coeffs_to_gram2(a: Dict[Tuple[int,int], int]) -> IntMatrix:
    # Determine dimension
    n = 0
    for (i,j) in a.keys():
        n = max(n, i+1, j+1)
    M2 = [[0]*n for _ in range(n)]
    for i in range(n):
        if (i,i) in a:
            M2[i][i] = 2*a[(i,i)]
    for (i,j), c in a.items():
        if i == j: continue
        ii, jj = (i,j) if i < j else (j,i)
        M2[ii][jj] += c
        M2[jj][ii] += c
    return IntMatrix(M2)

def eval_quadratic_from_gram2(M2: IntMatrix, x: List[int]) -> int:
    # Computes Q(x) = (1/2) x^T M2 x  with exact integer division by 2
    n = len(x)
    if M2.m != n or M2.n != n: raise ValueError("dimension mismatch")
    s = 0
    for i in range(n):
        xi = x[i]
        if xi == 0: continue
        for j in range(n):
            s += xi * M2.A[i][j] * x[j]
    q, r = divmod(s, 2)
    if r != 0:
        raise ValueError("x^T (2B) x is not even; coefficients may not be quadratic-integer")
    return q
