(rules:rule_6)=
# Rule 6: Implement Test-Driven Development with AI

Frame your test requirements as behavioral specifications before requesting implementation code, and tell the AI what success looks like through concrete test cases. This test-first approach forces you to articulate edge cases, expected inputs/outputs, and failure modes that might otherwise be overlooked. The AI responds better to specific test scenarios than vague functionality descriptions. By providing comprehensive test specifications, you guide the AI toward more robust, production-ready implementations. AI tools (such as chatbots or Github's Spec Kit) can help develop these specifications in a way that will optimally guide the model. Keep a close eye on the tests that are generated, since the models will often modify the tests to pass without actually solving the problem rather than generating suitable code. Be especially aware that coding agents may, intentionally or not, generate placeholder data or mock implementations that merely satisfy the test structure without validating actual logic. In many cases, the AI may insert fabricated input values or dummy functions that appear to meet acceptance criteria but do not reflect true functionality. These "paper tests" can be dangerously misleading, seemingly passing as tests while masking broken or incomplete logic. In addition, whenever a bug is identified during your development cycle, ask the model to generate a test that catches the bug, to ensure that it's not re-introduced in the future.

## What separates positive from flawed examples

Flawed examples ask for implementation first and maybe add tests later as an afterthought, testing only the happy path and perhaps one or two edge cases you thought of. You get code that technically works for anticipated uses but fails on boundary conditions, numerical edge cases, and subtle failure modes you didn't consider. When bugs appear, you patch the code without adding tests, so the same bugs reappear later. The AI often modifies tests to make them pass rather than fixing the actual problem, or generates "paper tests" with placeholder data and mock implementations that merely satisfy test structure without validating actual logic.

Positive examples start with tests that specify expected behavior before requesting any implementation. You articulate success criteria through concrete test cases, leveraging AI to systematically identify potential failure modes, edge cases, boundary conditions, and scenarios where numerical instability could occur. You evaluate these AI-generated test suggestions critically; some won't be relevant, but many reveal real gaps in your testing strategy. When the AI generates code, you can immediately verify it meets your specifications. When bugs appear, you first write a test that catches the bug, then fix the implementation, watching carefully to ensure the AI doesn't modify tests to make them pass. Your test suite becomes comprehensive and your code robust against inputs you didn't anticipate.

---

### Example 1: Implementation First, Tests as Afterthought

The user asks for code without specifying what success looks like. The AI generates something that works for basic cases but has no clear specification. When tests are added later, they just verify what the code currently does rather than what it should do. Edge cases are discovered in production. When bugs appear, the code gets patched without adding tests to prevent regression. The cycle repeats.

````{prf:example} Code without behavioral specification
:label: rule6-bad-ex1
:class: dropdown

**User Prompt:**
```
Create a function to threshold brain connectivity matrices.
```

**AI Response:**
```python
def threshold_connectivity(matrix, threshold=0.5):
    """Threshold connectivity matrix."""
    thresholded = matrix.copy()
    thresholded[matrix < threshold] = 0
    return thresholded
```

**Later, user adds tests:**
```python
def test_threshold():
    matrix = np.array([[0.8, 0.3], [0.3, 0.9]])
    result = threshold_connectivity(matrix, threshold=0.5)
    assert result[0, 0] == 0.8
    assert result[0, 1] == 0.0
```

**Production Bug Discovered:**
```
User: "It's zeroing the diagonal! Brain regions are perfectly correlated with 
themselves, we need to preserve the diagonal."

AI: "Let me fix that..."
[Patches code without adding test for diagonal preservation]

[Later, another bug: negative correlations being zeroed]
[Another patch without test]
[Cycle continues...]
```
````

---

### Example 2: Tests Define Behavior First

The user specifies expected behavior through comprehensive test cases before asking for implementation. The tests cover happy path, edge cases, error conditions, and domain-specific requirements (like preserving the diagonal). The AI now has a clear specification of what success looks like. The implementation naturally handles all specified cases. When bugs appear later, tests are added first to catch the bug, then the implementation is fixed.

````{prf:example} Behavioral specification through tests
:label: rule6-good-ex1
:class: dropdown

**User Prompt:**
```
I need a function to threshold brain connectivity matrices. Here's what success 
looks like through test cases:

def test_basic_thresholding():
    """Values below threshold should be zeroed."""
    matrix = np.array([[0.8, 0.3], [0.3, 0.9]])
    result = threshold_connectivity(matrix, threshold=0.5)
    assert result[0, 0] == 0.8  # Above threshold preserved
    assert result[0, 1] == 0.0  # Below threshold zeroed
    assert result[1, 1] == 0.9

def test_preserve_diagonal():
    """Diagonal should always be preserved (self-correlation = 1)."""
    matrix = np.array([[1.0, 0.3], [0.3, 1.0]])
    result = threshold_connectivity(matrix, threshold=0.5)
    assert result[0, 0] == 1.0  # Diagonal preserved even if would be zeroed
    assert result[1, 1] == 1.0

def test_negative_correlations():
    """Negative correlations should be preserved if above absolute threshold."""
    matrix = np.array([[1.0, -0.7], [-0.7, 1.0]])
    result = threshold_connectivity(matrix, threshold=0.5)
    assert result[0, 1] == -0.7  # Strong negative preserved
    assert result[1, 0] == -0.7

def test_weak_negative_zeroed():
    """Weak negative correlations should be zeroed."""
    matrix = np.array([[1.0, -0.2], [-0.2, 1.0]])
    result = threshold_connectivity(matrix, threshold=0.5)
    assert result[0, 1] == 0.0
    assert result[1, 0] == 0.0

def test_symmetry_preserved():
    """Thresholding should preserve matrix symmetry."""
    matrix = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.7], [0.3, 0.7, 1.0]])
    result = threshold_connectivity(matrix, threshold=0.5)
    assert np.allclose(result, result.T)  # Still symmetric

def test_input_validation():
    """Should reject invalid inputs."""
    with pytest.raises(ValueError):
        threshold_connectivity(np.array([1, 2, 3]))  # Not 2D
    
    with pytest.raises(ValueError):
        threshold_connectivity(np.array([[1, 2], [3, 4]]))  # Not symmetric

Now implement threshold_connectivity() to pass all these tests.
```

**AI Response:**
```python
def threshold_connectivity(matrix, threshold=0.5):
    """
    Threshold connectivity matrix while preserving key properties.
    
    Parameters
    ----------
    matrix : np.ndarray
        Symmetric connectivity matrix
    threshold : float
        Threshold value (applied to absolute values)
        
    Returns
    -------
    np.ndarray
        Thresholded matrix with diagonal and symmetry preserved
    """
    # Validation
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Matrix must be symmetric")
    
    # Threshold by absolute value, preserving sign
    thresholded = matrix.copy()
    mask = np.abs(matrix) < threshold
    
    # Zero values below threshold, but preserve diagonal
    np.fill_diagonal(mask, False)
    thresholded[mask] = 0
    
    return thresholded
```
````

---

### Example 3: Test-First Bug Prevention

A bug is discovered in production. Instead of immediately patching the code, the user first writes a test that catches the bug. This ensures the bug won't be reintroduced later. Then the implementation is fixed to pass the new test. The test suite grows to cover real-world failure modes. Each bug becomes a permanent regression test.

````{prf:example} Bug found, test added first, then fix
:label: rule6-good-ex2
:class: dropdown

**Production Bug Report:**
```
User: "The connectivity thresholding is failing when the matrix contains NaN 
values (which happens when some brain regions were censored during preprocessing). 
It's not handling NaN correctly."
```

**User Response:**
```
First, let's write a test that catches this bug:

def test_nan_handling():
    """NaN values should be preserved through thresholding."""
    matrix = np.array([[1.0, 0.6, np.nan], 
                       [0.6, 1.0, 0.3], 
                       [np.nan, 0.3, 1.0]])
    result = threshold_connectivity(matrix, threshold=0.5)
    
    # NaN should be preserved
    assert np.isnan(result[0, 2])
    assert np.isnan(result[2, 0])
    
    # Valid values should be thresholded normally
    assert result[0, 1] == 0.6  # Above threshold
    assert result[1, 2] == 0.0  # Below threshold
    
    # Diagonal preserved
    assert result[0, 0] == 1.0
    assert result[1, 1] == 1.0
    assert result[2, 2] == 1.0

Run this test with the current implementation - it should fail, confirming the bug.

Then fix threshold_connectivity() to pass this test.
```

**AI verifies test fails, then fixes:**
```python
def threshold_connectivity(matrix, threshold=0.5):
    """Threshold connectivity matrix while preserving key properties."""
    # Validation
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")
    if not np.allclose(matrix, matrix.T, equal_nan=True):  # Allow NaN
        raise ValueError("Matrix must be symmetric")
    
    thresholded = matrix.copy()
    
    # Create mask for values below threshold, excluding NaN
    mask = (np.abs(matrix) < threshold) & ~np.isnan(matrix)
    
    # Preserve diagonal
    np.fill_diagonal(mask, False)
    
    # Zero values below threshold
    thresholded[mask] = 0
    
    return thresholded
```

**Result:**
```
Bug is fixed, and we now have a permanent test that prevents this bug from 
being reintroduced. The test suite is growing to cover real-world edge cases.
```
````

---

### Example 4: Catching AI Test Manipulation

The user provides test specifications, but the AI modifies the tests to make them pass rather than fixing the implementation. The user catches this by carefully reviewing what changed. They explicitly instruct the AI to not modify tests and to fix the implementation instead. This prevents the AI from taking the easy path of making tests less strict.

````{prf:example} AI tries to modify tests, user catches it
:label: rule6-good-ex3
:class: dropdown

**User Prompt:**
```
Implement normalize_timeseries() to pass these tests:

def test_zero_mean():
    """Normalized timeseries should have zero mean."""
    ts = np.array([1, 2, 3, 4, 5])
    normalized = normalize_timeseries(ts)
    assert np.abs(np.mean(normalized)) < 1e-10

def test_unit_variance():
    """Normalized timeseries should have unit variance."""
    ts = np.array([1, 2, 3, 4, 5])
    normalized = normalize_timeseries(ts)
    assert np.abs(np.var(normalized) - 1.0) < 1e-10

def test_constant_timeseries():
    """Constant timeseries should return zeros (can't normalize)."""
    ts = np.array([5, 5, 5, 5, 5])
    normalized = normalize_timeseries(ts)
    assert np.allclose(normalized, 0)
```

**AI Response (WRONG APPROACH):**
```python
def normalize_timeseries(ts):
    """Normalize timeseries to zero mean and unit variance."""
    # BUG: Using sample variance (ddof=1) instead of population variance (ddof=0)
    # This means var(normalized) will be slightly > 1.0
    normalized = (ts - np.mean(ts)) / np.std(ts, ddof=1)
    return normalized

# AI also modified the test to hide the bug:
def test_unit_variance():
    """Normalized timeseries should have approximately unit variance."""
    ts = np.array([1, 2, 3, 4, 5])
    normalized = normalize_timeseries(ts)
    assert np.abs(np.var(normalized) - 1.0) < 0.1  # CHANGED: was < 1e-10
```

**User:**
```
STOP. You modified my test to make it less strict. Do NOT change the tests. 
The tests define the specification.

The original tolerance (< 1e-10) is correct. Your implementation should pass 
that strict test. The issue is you're using ddof=1 in np.std(), which gives
sample standard deviation. For normalization to unit variance, you need
population standard deviation (ddof=0, the default).

Keep the tests EXACTLY as I specified them and fix the implementation.
```

**AI Response (CORRECTED):**
```python
def normalize_timeseries(ts):
    """
    Normalize timeseries to zero mean and unit variance.
    
    For constant timeseries (std=0), returns zeros.
    """
    std = np.std(ts)
    if std == 0:
        return np.zeros_like(ts)
    
    normalized = (ts - np.mean(ts)) / std  # ddof=0 (default)
    return normalized

# Tests remain UNCHANGED from original specification
def test_zero_mean():
    """Normalized timeseries should have zero mean."""
    ts = np.array([1, 2, 3, 4, 5])
    normalized = normalize_timeseries(ts)
    assert np.abs(np.mean(normalized)) < 1e-10  # Original spec

def test_unit_variance():
    """Normalized timeseries should have unit variance."""
    ts = np.array([1, 2, 3, 4, 5])
    normalized = normalize_timeseries(ts)
    assert np.abs(np.var(normalized) - 1.0) < 1e-10  # Original spec

def test_constant_timeseries():
    """Constant timeseries should return zeros (can't normalize)."""
    ts = np.array([5, 5, 5, 5, 5])
    normalized = normalize_timeseries(ts)
    assert np.allclose(normalized, 0)  # Original spec
```
````
