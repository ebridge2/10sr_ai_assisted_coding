(rules:rule_9)=
# Rule 9: Use AI to Refine and Validate Tests

AI is exceptionally good at identifying edge cases you might miss and suggesting comprehensive test scenarios. Feed it your function and ask it to generate tests for boundary conditions, type validation, error handling, and numerical stability. Ask it what sorts of problems your code might experience issues with, within your specified API bounds, and why those might (or might not) be relevant to address. AI can help you move beyond happy-path testing to robust validation that includes malformed inputs, extreme values, and race conditions. Use AI to review your existing tests and identify gaps in coverage or scenarios you haven't considered.

## What separates positive from flawed examples

Flawed examples only test the happy path and maybe one or two edge cases you thought of. You miss boundary conditions, numerical edge cases, and subtle failure modes. Your tests pass but the code still breaks in production on inputs you didn't anticipate. You don't systematically explore what could go wrong.

Positive examples leverage AI to systematically identify potential failure modes. You ask what edge cases exist, what boundary conditions matter, where numerical instability could occur. The AI suggests test scenarios you wouldn't have thought of. You evaluate these suggestions critically (some won't be relevant), but many reveal real gaps in your testing. Your test suite becomes much more comprehensive.

---

## Example 1: Only Happy Path Testing

The user writes basic tests that verify the function works for normal inputs. Edge cases, boundary conditions, and failure modes are completely unexplored. The tests pass, giving false confidence. Then production encounters inputs the tests never covered. The code fails in ways that could have been caught with more comprehensive testing.

````{prf:example} Minimal testing without edge case exploration
:label: rule9-bad-ex1
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

## Example 2: AI-Assisted Comprehensive Test Generation

The user asks AI to systematically identify edge cases and potential failure modes. The AI suggests boundary conditions, numerical edge cases, error scenarios, and performance considerations the user hadn't thought of. The user evaluates each suggestion and implements the relevant ones. The resulting test suite is much more robust and catches real issues before production.

````{prf:example} Systematic edge case identification with AI
:label: rule9-good-ex1
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

## Example 3: AI Identifies Performance Edge Cases

The user asks AI to identify performance-related edge cases and potential bottlenecks. The AI suggests benchmark tests for different input sizes and parallelization issues. This catches performance regressions before they reach production.

````{prf:example} Performance and scalability testing
:label: rule9-good-ex2
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
