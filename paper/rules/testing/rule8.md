(rules:rule_8)=
# Rule 8: Build Comprehensive Testing Infrastructure Early

Early in your development cycle, prioritize building comprehensive testing infrastructure including automated validation workflows. In addition to the standard benefits of having testing infrastructure, this is particularly relevant when coding with AI, as it is very easy to miss minor changes in code made by the AI coding assistant that might have substantial downstream ramifications.

AI excels at generating the boilerplate for sophisticated tools (such as GitHub Actions, pre-commit hooks, and test orchestration) that ensure your code is validated on every push. Ask it to create testing workflows that include unit tests, integration tests, performance benchmarks, and even data validation checks. This infrastructure investment pays dividends by catching inconsistencies before they reach production and maintaining code quality across collaborators. The AI can help you implement sophisticated testing patterns like parameterized tests, fixtures, and mocking that would be tedious to write from scratch.

## What separates positive from flawed examples

Flawed examples skip testing infrastructure in favor of writing more features. Tests (if they exist) are run manually and inconsistently. Changes slip through without validation. By the time you discover problems, they've compounded into major issues. With AI generating code quickly, technical debt accumulates faster than you can track it.

Positive examples invest in comprehensive testing infrastructure from day one. Every push triggers automated tests. Pre-commit hooks catch issues before they enter the codebase. The infrastructure grows alongside the code. Because AI can generate code so quickly, the testing infrastructure prevents that speed from becoming a liability.

---

## Example 1: No Testing Infrastructure

The project has some tests but they're run manually when someone remembers. No CI/CD pipeline. No pre-commit hooks. No automated validation. The AI generates code changes quickly, but there's no systematic verification. Small issues accumulate into major problems. When bugs are discovered, it's unclear when they were introduced. Code quality degrades as the team scales.

````{prf:example} Manual testing only
:label: rule8-bad-ex1
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

## Example 2: Comprehensive Testing Infrastructure

The project has a complete testing infrastructure set up early. GitHub Actions runs tests on every push. Pre-commit hooks catch issues before commit. Multiple test suites (unit, integration, performance) run automatically. The AI helped generate most of this boilerplate. Now when the AI generates code changes, they're automatically validated. Issues are caught immediately with clear error messages pointing to the problem.

````{prf:example} Full CI/CD with automated validation
:label: rule8-good-ex1
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

## Example 3: Parameterized Tests and Fixtures

The user asks AI to create sophisticated test patterns that would be tedious to write manually. Parameterized tests cover multiple input combinations. Fixtures provide reusable test data. The infrastructure makes it easy to add new test cases. This catches edge cases that manual testing would miss.

````{prf:example} Advanced testing patterns generated by AI
:label: rule8-good-ex2
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

---

**Key Takeaways:**
- Build testing infrastructure early, before it becomes painful
- Use AI to generate CI/CD pipelines, pre-commit hooks, test organization
- Automate everything: tests should run on every push
- Include multiple test types: unit, integration, performance
- Use fixtures and parameterization for thorough coverage
- Track coverage and enforce minimum thresholds
- Infrastructure prevents AI-generated code from accumulating technical debt
- The faster you can generate code, the more critical automated testing becomes