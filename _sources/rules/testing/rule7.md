(rules:rule_7)=
# Rule 7: Leverage AI for Test Planning and Refinement

AI is exceptionally good at identifying edge cases you might miss and suggesting comprehensive test scenarios. Feed it your function and ask it to generate tests for boundary conditions, type validation, error handling, and numerical stability. Ask it what sorts of problems your code might experience issues with, within your specified API bounds, and why those might (or might not) be relevant to address. AI can help you move beyond testing only expected behavior to robust validation that includes malformed inputs, extreme values, and unexpected conditions. Additionally, you can use AI to review your existing tests and identify gaps in coverage or scenarios you haven't considered. The AI can help you implement sophisticated testing patterns like parameterized tests, fixtures, and mocking that would be tedious to write from scratch. If you anticipate having future collaborators for your project, you may find it helpful to prioritize building testing infrastructure early. This often includes automated validation workflows, wherein you are able to test your code automatically as you integrate changes into the broader project. AI excels at generating the boilerplate for many of these sophisticated testing tools (such as GitHub Actions, pre-commit hooks, and test orchestration) that ensure your code is validated on every push. 

## What separates positive from flawed examples

Flawed examples skip testing infrastructure in favor of writing more features. Tests (if they exist) are run manually and inconsistently. Changes slip through without validation. By the time you discover problems, they've compounded into major issues. With AI generating code quickly, technical debt accumulates faster than you can track it.

Positive examples invest in comprehensive testing infrastructure from day one. Every push triggers automated tests. Pre-commit hooks catch issues before they enter the codebase. The infrastructure grows alongside the code. Because AI can generate code so quickly, the testing infrastructure prevents that speed from becoming a liability.

---

### Example 1: Only Happy Path Testing

The user writes basic tests that verify the function works for normal inputs. Edge cases, boundary conditions, and failure modes are completely unexplored. The tests pass, giving false confidence. Then production encounters inputs the tests never covered. The code fails in ways that could have been caught with more comprehensive testing.

````{prf:example} Minimal testing without edge case exploration
:label: rule7-bad-ex1
:class: dropdown

**Current Tests:**
```python
def test_compute_correlation():
    """Test basic correlation computation."""
    data = np.random.randn(10, 100)
    result = compute_correlation(data)
    assert result.shape == (10, 10)
    assert np.allclose(np.diag(result), 1.0)

def test_different_sizes():
    """Test with different matrix sizes."""
    data = np.random.randn(50, 200)
    result = compute_correlation(data)
    assert result.shape == (50, 50)
```

**Production Failures (that tests didn't catch):**
```
- Crashes with single region (1, n) input
- Returns NaN for constant timeseries
- Numerical instability with highly correlated data
- Fails silently with infinite values
- Wrong results when timepoints < regions
- Memory issues with very large matrices
- Thread-safety problems in parallel processing
```
````

---


### Example 2: No Testing Infrastructure

The project has some tests but they're run manually when someone remembers. No CI/CD pipeline. No pre-commit hooks. No automated validation. The AI generates code changes quickly, but there's no systematic verification. Small issues accumulate into major problems. When bugs are discovered, it's unclear when they were introduced. Code quality degrades as the team scales.

````{prf:example} Manual testing only
:label: rule7-bad-ex2
:class: dropdown

**Project State:**
```
project/
├── src/
│   └── analysis.py  # Main code
├── tests/
│   └── test_analysis.py  # Some tests, rarely run
└── README.md

# No CI/CD
# No pre-commit hooks
# No automated validation
# Tests run manually with: python -m pytest (when remembered)
```

**Development Process:**
```
Developer 1: Makes changes with AI assistance
Developer 2: Also makes changes with AI assistance
[Neither runs tests]
[Changes get merged]
[Week later: production breaks]

Team: "When was this bug introduced?"
[No automated testing means no clear record]
[Have to manually bisect commits]
[Lost days of productivity]
```
````

---

### Example 3: AI-Assisted Comprehensive Test Generation

The user asks AI to systematically identify edge cases and potential failure modes. The AI suggests boundary conditions, numerical edge cases, error scenarios, and performance considerations the user hadn't thought of. The user evaluates each suggestion and implements the relevant ones. The resulting test suite is much more robust and catches real issues before production.

````{prf:example} Systematic edge case identification with AI
:label: rule7-good-ex1
:class: dropdown

**User Prompt:**
```
Here's my correlation function:

def compute_correlation(data, method='pearson'):
    """
    Compute correlation matrix from timeseries data.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_regions, n_timepoints)
        Timeseries data for each region
    method : str
        Correlation method ('pearson' or 'spearman')
    
    Returns
    -------
    np.ndarray, shape (n_regions, n_regions)
        Correlation matrix
    """
    if method == 'pearson':
        return np.corrcoef(data)
    elif method == 'spearman':
        from scipy.stats import spearmanr
        return spearmanr(data, axis=1).correlation
    else:
        raise ValueError(f"Unknown method: {method}")

I have basic tests for normal operation. Help me identify edge cases and 
potential failure modes I should test. For each, explain why it matters and 
whether it's relevant to my use case (neuroimaging analysis).
```

**AI Response:**
```
Here are systematic edge cases and failure modes to consider:

**BOUNDARY CONDITIONS:**

1. Single region (n_regions=1):
   - Result shape should be (1,1) with value 1.0
   - Relevant: Yes, sometimes analyzing single ROI
   - Test: Verify doesn't crash, returns [[1.0]]

2. Two timepoints (minimum for correlation):
   - Correlation will be +/-1.0 (perfectly determined)
   - Relevant: Yes, heavily censored data might have few timepoints
   - Test: Verify handles gracefully, results are +/-1.0

3. Single timepoint (invalid):
   - Cannot compute correlation with one sample
   - Relevant: Yes, edge case of censoring
   - Test: Should raise ValueError with clear message

**NUMERICAL EDGE CASES:**

4. Constant timeseries (zero variance):
   - Division by zero in correlation formula
   - Relevant: Very yes - motion artifacts can create constant periods
   - Test: Should return NaN or handle gracefully, not crash
   - Recommendation: Document this behavior explicitly

5. Near-constant timeseries (numerical instability):
   - Very small variance can cause numerical issues
   - Relevant: Yes, some regions have very low signal
   - Test: Variance < 1e-10, verify results are stable

6. Infinite or NaN values:
   - Can arise from preprocessing errors
   - Relevant: Very yes - common in real data
   - Test: Should either handle or raise clear error
   - Recommendation: Add explicit NaN handling

7. Very large values (overflow risk):
   - Could cause numerical overflow in intermediate calculations
   - Relevant: Moderate - unusual but possible with bad preprocessing
   - Test: Values ~1e100, verify no overflow

8. Highly correlated regions (r ≈ 1.0):
   - Can cause numerical instability in some algorithms
   - Relevant: Yes - anatomically adjacent regions are highly correlated
   - Test: Synthetic data with r > 0.9999, verify stability

**DATA STRUCTURE ISSUES:**

9. Wrong number of dimensions:
   - Function expects 2D, might receive 1D or 3D
   - Relevant: Yes - common user error
   - Test: Should raise ValueError with helpful message

10. n_timepoints < n_regions:
    - Technically valid but often indicates mistake
    - Relevant: Yes - common error (transposed data)
    - Test: Should work correctly or warn user

11. Non-contiguous array:
    - Can cause unexpected behavior with numpy operations
    - Relevant: Low - usually not an issue
    - Test: Verify works with data.T.copy() (non-contiguous)

**TYPE VALIDATION:**

12. Wrong dtype (integers, booleans):
    - Should handle or convert appropriately
    - Relevant: Moderate - might receive integer data
    - Test: Integer input should work or raise clear error

13. Wrong input type (list, DataFrame):
    - Should require numpy array
    - Relevant: Moderate - common user error
    - Test: Should raise TypeError with helpful message

**PERFORMANCE CONSIDERATIONS:**

14. Very large matrices (memory):
    - n_regions = 10,000 creates 100M element output
    - Relevant: Low for typical neuroimaging (n_regions < 1000)
    - Test: If supporting large matrices, verify memory usage

15. Thread safety (parallel processing):
    - If used in parallel pipelines
    - Relevant: High - often process multiple subjects in parallel
    - Test: Verify no race conditions with concurrent calls

**STATISTICAL PROPERTIES:**

16. Negative correlations:
    - Should preserve sign
    - Relevant: Yes - anti-correlations are meaningful
    - Test: Synthetic anticorrelated data, verify negative values

17. Diagonal values:
    - Should always be 1.0 (self-correlation)
    - Relevant: Yes - fundamental property
    - Test: Always verify diagonal is 1.0

18. Symmetry:
    - Result should be symmetric matrix
    - Relevant: Yes - fundamental property
    - Test: Always verify result == result.T
```

**User Response:**
```
Great analysis. Let me implement tests for the high-relevance cases:
- Single region, two timepoints, single timepoint (boundaries)
- Constant and near-constant timeseries (numerical)
- NaN/infinite values (data quality)
- Wrong dimensions (user error)
- Negative correlations, diagonal, symmetry (statistical properties)

Skip the low-relevance ones (overflow, very large matrices).

Generate comprehensive tests for these cases.
```

**AI generates comprehensive test suite:**
```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestComputeCorrelation:
    """Comprehensive tests for compute_correlation function."""
    
    # Boundary conditions
    def test_single_region(self):
        """Single region should return 1x1 matrix with value 1.0."""
        data = np.random.randn(1, 100)
        result = compute_correlation(data)
        assert result.shape == (1, 1)
        assert_allclose(result, [[1.0]])
    
    def test_minimum_timepoints(self):
        """Two timepoints should give correlation of +/-1.0."""
        data = np.array([[1, 2], [2, 4], [1, 1]])
        result = compute_correlation(data)
        assert result.shape == (3, 3)
        # Perfect correlations or anticorrelations
        assert np.all(np.abs(result) == 1.0)
    
    def test_single_timepoint_fails(self):
        """Single timepoint should raise clear error."""
        data = np.array([[1], [2], [3]])
        with pytest.raises(ValueError, match="at least 2"):
            compute_correlation(data)
    
    # Numerical edge cases
    def test_constant_timeseries(self):
        """Constant timeseries should return NaN correlation."""
        data = np.ones((3, 100))
        data[1, :] = 2  # Different constant
        result = compute_correlation(data)
        # Self-correlations should be 1
        assert_allclose(np.diag(result), 1.0)
        # Cross-correlations undefined (NaN)
        assert np.isnan(result[0, 1])
        assert np.isnan(result[0, 2])
    
    def test_near_constant_numerically_stable(self):
        """Near-constant timeseries should be numerically stable."""
        data = np.ones((2, 100))
        # Add tiny noise
        data += np.random.randn(2, 100) * 1e-12
        result = compute_correlation(data)
        # Should not crash, results should be finite
        assert np.all(np.isfinite(result))
    
    def test_nan_values(self):
        """NaN values should be handled or raise clear error."""
        data = np.random.randn(3, 100)
        data[0, 50] = np.nan
        # Current implementation propagates NaN
        result = compute_correlation(data)
        assert np.isnan(result[0, :]).any()
        # Or could require: with pytest.raises(ValueError, match="NaN")
    
    def test_infinite_values(self):
        """Infinite values should raise clear error."""
        data = np.random.randn(3, 100)
        data[0, 50] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            compute_correlation(data)
    
    def test_highly_correlated_stable(self):
        """Very high correlations should be numerically stable."""
        # Create nearly identical timeseries
        base = np.random.randn(100)
        data = np.array([base, base + 1e-10 * np.random.randn(100)])
        result = compute_correlation(data)
        # Should be very close to 1.0, not NaN or unstable
        assert 0.999 < result[0, 1] < 1.0
    
    # Data structure issues
    def test_wrong_dimensions_1d(self):
        """1D array should raise clear error."""
        data = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="2D.*got 1D"):
            compute_correlation(data)
    
    def test_wrong_dimensions_3d(self):
        """3D array should raise clear error."""
        data = np.random.randn(10, 20, 30)
        with pytest.raises(ValueError, match="2D.*got 3D"):
            compute_correlation(data)
    
    def test_transposed_warning(self):
        """More timepoints than regions should work but warn."""
        # Common mistake: (n_timepoints, n_regions) instead of (n_regions, n_timepoints)
        data = np.random.randn(5, 100)  # Correct
        result1 = compute_correlation(data)
        
        data_transposed = np.random.randn(100, 5)  # Likely wrong
        result2 = compute_correlation(data_transposed)
        
        assert result1.shape == (5, 5)
        assert result2.shape == (100, 100)  # Works but probably not intended
    
    # Statistical properties
    def test_negative_correlations_preserved(self):
        """Negative correlations should be preserved."""
        # Create anticorrelated timeseries
        base = np.random.randn(100)
        data = np.array([base, -base, base * 2])
        result = compute_correlation(data)
        # First two should be perfectly anticorrelated
        assert_allclose(result[0, 1], -1.0, atol=1e-10)
        assert_allclose(result[1, 0], -1.0, atol=1e-10)
    
    def test_diagonal_always_one(self):
        """Diagonal should always be 1.0 (self-correlation)."""
        data = np.random.randn(20, 100)
        result = compute_correlation(data)
        assert_allclose(np.diag(result), 1.0, atol=1e-10)
    
    def test_symmetry(self):
        """Result should be symmetric matrix."""
        data = np.random.randn(10, 100)
        result = compute_correlation(data)
        assert_allclose(result, result.T, atol=1e-10)
    
    def test_values_in_range(self):
        """All correlation values should be in [-1, 1]."""
        data = np.random.randn(20, 100)
        result = compute_correlation(data)
        assert np.all(result >= -1.0001)  # Small tolerance for numerical error
        assert np.all(result <= 1.0001)
```
````

---

### Example 4: AI Identifies Performance Edge Cases

The user asks AI to identify performance-related edge cases and potential bottlenecks. The AI suggests benchmark tests for different input sizes and parallelization issues. This catches performance regressions before they reach production.

````{prf:example} Performance and scalability testing
:label: rule7-good-ex2
:class: dropdown

**User Prompt:**
```
My compute_correlation function works correctly but I'm worried about performance.
What edge cases should I test for performance and scalability?

Typical use: 100-1000 brain regions, 200-1000 timepoints.
Sometimes process 100+ subjects in parallel.
```

**AI Response:**
```
Performance edge cases to test:

1. **Scaling with n_regions:**
   - Small (10 regions): baseline
   - Medium (100 regions): typical case
   - Large (1000 regions): upper bound
   - Very large (5000 regions): stress test
   - Expectation: O(n²) scaling for output size

2. **Scaling with n_timepoints:**
   - Few (50 timepoints): minimum useful
   - Typical (500 timepoints): common case  
   - Many (2000 timepoints): large study
   - Expectation: Linear scaling with timepoints

3. **Memory usage:**
   - Track peak memory for large matrices
   - 1000 regions = 8MB output (double precision)
   - Should not create unnecessary intermediate copies

4. **Parallel safety:**
   - Concurrent calls from multiple threads
   - Verify no race conditions
   - Check for numpy thread safety issues

5. **Different methods performance:**
   - Pearson vs Spearman speed comparison
   - Spearman is typically slower (ranking required)

Here are benchmark tests:
```

```python
import pytest
from memory_profiler import memory_usage

@pytest.mark.benchmark(group="scaling_regions")
@pytest.mark.parametrize("n_regions", [10, 100, 500, 1000])
def test_performance_regions(benchmark, n_regions):
    """Test performance scaling with number of regions."""
    data = np.random.randn(n_regions, 500)
    
    result = benchmark(compute_correlation, data)
    
    # Performance expectations based on n_regions
    if n_regions <= 100:
        assert benchmark.stats['mean'] < 0.1  # Very fast
    elif n_regions <= 500:
        assert benchmark.stats['mean'] < 0.5  # Still fast
    else:
        assert benchmark.stats['mean'] < 2.0  # Acceptable

@pytest.mark.benchmark(group="scaling_timepoints")
@pytest.mark.parametrize("n_timepoints", [50, 200, 500, 1000, 2000])
def test_performance_timepoints(benchmark, n_timepoints):
    """Test performance scaling with number of timepoints."""
    data = np.random.randn(100, n_timepoints)
    
    result = benchmark(compute_correlation, data)
    
    # Should scale roughly linearly with timepoints
    # All should be fast for 100 regions
    assert benchmark.stats['mean'] < 0.5

def test_memory_usage_large():
    """Test memory usage doesn't explode for large matrices."""
    def compute_large():
        data = np.random.randn(1000, 1000)
        result = compute_correlation(data)
        return result
    
    mem_usage = memory_usage(compute_large)
    peak_mb = max(mem_usage)
    
    # Should not use more than ~100MB for this size
    # (1000x1000 input = 8MB, output = 8MB, overhead should be modest)
    assert peak_mb < 100

def test_parallel_safety():
    """Test thread safety for parallel processing."""
    from concurrent.futures import ThreadPoolExecutor
    import threading
    
    def compute_random():
        data = np.random.randn(100, 500)
        result = compute_correlation(data)
        # Verify result is valid
        assert result.shape == (100, 100)
        assert np.allclose(np.diag(result), 1.0)
        return result
    
    # Run 20 concurrent computations
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(compute_random) for _ in range(20)]
        results = [f.result() for f in futures]
    
    # All should complete successfully
    assert len(results) == 20

@pytest.mark.parametrize("method", ['pearson', 'spearman'])
def test_method_performance(benchmark, method):
    """Compare performance of different correlation methods."""
    data = np.random.randn(100, 500)
    
    result = benchmark(compute_correlation, data, method=method)
    
    if method == 'pearson':
        # Pearson should be fast
        assert benchmark.stats['mean'] < 0.1
    else:  # spearman
        # Spearman slower due to ranking, but not too slow
        assert benchmark.stats['mean'] < 1.0
```
````

---

### Example 5: Comprehensive Testing Infrastructure

The project has a complete testing infrastructure set up early. GitHub Actions runs tests on every push. Pre-commit hooks catch issues before commit. Multiple test suites (unit, integration, performance) run automatically. The AI helped generate most of this boilerplate. Now when the AI generates code changes, they're automatically validated. Issues are caught immediately with clear error messages pointing to the problem.

````{prf:example} Full CI/CD with automated validation
:label: rule7-good-ex3
:class: dropdown

**User Prompt:**
```
Set up comprehensive testing infrastructure for our neuroimaging analysis project.

Current structure:
src/
  neuroimaging/
    loader.py
    preprocessing.py
    analysis.py

We need:
1. GitHub Actions for CI/CD
2. Pre-commit hooks for code quality
3. Test organization (unit, integration, performance)
4. Coverage reporting
5. Automated dependency checking

Create all necessary config files and directory structure.
```

**AI Response (creates complete infrastructure):**

```yaml
# .github/workflows/tests.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with ruff
      run: ruff check src/ tests/
    
    - name: Type check with mypy
      run: mypy src/
    
    - name: Run unit tests
      run: pytest tests/unit/ -v --cov=neuroimaging --cov-report=xml
    
    - name: Run integration tests
      run: pytest tests/integration/ -v
    
    - name: Run performance benchmarks
      run: pytest tests/benchmarks/ --benchmark-only
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      if: matrix.python-version == '3.10'
```

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.10.0
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=1000]
```

```python
# tests/conftest.py - Shared fixtures
import pytest
import numpy as np
import nibabel as nib
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture
def sample_fmri_data():
    """Generate synthetic 4D fMRI data."""
    data = np.random.randn(64, 64, 32, 100)
    affine = np.eye(4)
    return nib.Nifti1Image(data, affine)

@pytest.fixture
def sample_connectivity_matrix():
    """Generate sample connectivity matrix."""
    n_regions = 100
    matrix = np.random.randn(n_regions, n_regions)
    matrix = (matrix + matrix.T) / 2  # Make symmetric
    np.fill_diagonal(matrix, 1.0)
    return matrix
```

```toml
# pyproject.toml - Tool configuration
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--strict-markers",
    "--cov=neuroimaging",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-fail-under=80",
]

[tool.coverage.run]
source = ["src"]
omit = ["*/tests/*", "*/conftest.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
]

[tool.black]
line-length = 100
target-version = ['py39', 'py310', 'py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "W", "I", "N", "UP", "B", "A", "C4", "PT"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
```

**Project Structure:**
```
project/
├── .github/
│   └── workflows/
│       └── tests.yml
├── .pre-commit-config.yaml
├── pyproject.toml
├── src/
│   └── neuroimaging/
│       ├── loader.py
│       ├── preprocessing.py
│       └── analysis.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_loader.py
│   │   ├── test_preprocessing.py
│   │   └── test_analysis.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── benchmarks/
│       └── test_performance.py
├── requirements.txt
└── requirements-dev.txt
```

**Result:**
- Every push triggers automated tests
- Pre-commit hooks catch issues locally
- Code coverage tracked automatically
- Multiple Python versions tested
- Performance regressions detected
- Issues caught immediately with clear feedback
````

---

### Example 6: Parameterized Tests and Fixtures

The user asks AI to create sophisticated test patterns that would be tedious to write manually. Parameterized tests cover multiple input combinations. Fixtures provide reusable test data. The infrastructure makes it easy to add new test cases. This catches edge cases that manual testing would miss.

````{prf:example} Advanced testing patterns generated by AI
:label: rule7-good-ex4
:class: dropdown

**User Prompt:**
```
Create comprehensive parameterized tests for our connectivity thresholding function.

Function signature:
def threshold_connectivity(matrix: np.ndarray, threshold: float, 
                          method: str = 'absolute') -> np.ndarray

Methods:
- 'absolute': threshold by absolute value
- 'proportional': keep top X% of connections
- 'statistical': threshold by p-value

Create tests using pytest fixtures and parametrization to cover:
- Multiple matrix sizes (small, medium, large)
- Different threshold values
- All three methods
- Edge cases (empty matrices, all zeros, all ones)
- Invalid inputs
```

**AI Response:**
```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

# Fixtures for reusable test data
@pytest.fixture(params=[10, 100, 500])
def matrix_size(request):
    """Different matrix sizes for testing."""
    return request.param

@pytest.fixture
def random_connectivity_matrix(matrix_size):
    """Generate random symmetric connectivity matrix."""
    matrix = np.random.randn(matrix_size, matrix_size)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    return matrix

@pytest.fixture
def sparse_matrix(matrix_size):
    """Generate sparse connectivity matrix (80% zeros)."""
    matrix = np.random.randn(matrix_size, matrix_size)
    matrix = (matrix + matrix.T) / 2
    mask = np.random.rand(matrix_size, matrix_size) > 0.2
    matrix[mask] = 0
    np.fill_diagonal(matrix, 1.0)
    return matrix

# Parameterized tests
@pytest.mark.parametrize("threshold,expected_sparsity", [
    (0.1, 0.9),   # Low threshold, mostly preserved
    (0.5, 0.5),   # Medium threshold
    (0.9, 0.1),   # High threshold, mostly zeroed
])
def test_absolute_thresholding(random_connectivity_matrix, threshold, expected_sparsity):
    """Test absolute value thresholding."""
    result = threshold_connectivity(random_connectivity_matrix, 
                                   threshold, method='absolute')
    
    # Check sparsity roughly matches expectation
    actual_sparsity = np.sum(result == 0) / result.size
    assert abs(actual_sparsity - expected_sparsity) < 0.2
    
    # Verify threshold was applied correctly
    nonzero_values = np.abs(result[result != 0])
    assert np.all(nonzero_values >= threshold)

@pytest.mark.parametrize("method", ['absolute', 'proportional', 'statistical'])
def test_preserve_properties(random_connectivity_matrix, method):
    """All methods should preserve key matrix properties."""
    if method == 'absolute':
        result = threshold_connectivity(random_connectivity_matrix, 0.5, method=method)
    elif method == 'proportional':
        result = threshold_connectivity(random_connectivity_matrix, 0.2, method=method)
    else:  # statistical
        result = threshold_connectivity(random_connectivity_matrix, 0.05, method=method)
    
    # Diagonal preserved
    assert_allclose(np.diag(result), np.diag(random_connectivity_matrix))
    
    # Symmetry preserved
    assert_allclose(result, result.T)
    
    # Shape preserved
    assert result.shape == random_connectivity_matrix.shape

@pytest.mark.parametrize("invalid_input,error_type,error_match", [
    (np.array([1, 2, 3]), ValueError, "2D"),                    # 1D array
    (np.array([[1, 2], [3, 4]]), ValueError, "symmetric"),     # Not symmetric
    (np.random.randn(100, 200), ValueError, "square"),         # Not square
])
def test_input_validation(invalid_input, error_type, error_match):
    """Test error handling for invalid inputs."""
    with pytest.raises(error_type, match=error_match):
        threshold_connectivity(invalid_input, 0.5)

@pytest.mark.parametrize("method", ['absolute', 'proportional', 'statistical'])
def test_edge_case_all_zeros(method):
    """Handle matrix of all zeros (except diagonal)."""
    matrix = np.zeros((10, 10))
    np.fill_diagonal(matrix, 1.0)
    
    result = threshold_connectivity(matrix, 0.5, method=method)
    assert_allclose(result, matrix)  # Should return unchanged

@pytest.mark.parametrize("matrix_size", [10, 100, 1000])
@pytest.mark.benchmark
def test_performance_scaling(benchmark, matrix_size):
    """Test performance scales reasonably with matrix size."""
    matrix = np.random.randn(matrix_size, matrix_size)
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 1.0)
    
    result = benchmark(threshold_connectivity, matrix, 0.5, method='absolute')
    
    # Should complete in reasonable time even for large matrices
    assert benchmark.stats['mean'] < 1.0  # Less than 1 second
```
````
