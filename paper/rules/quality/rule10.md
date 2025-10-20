(rules:rule_10)=
# Rule 10: Refine Code Incrementally with Focused Objectives

Once you have working, tested code, resist the temptation to ask AI to "improve my codebase." Instead, approach refinement incrementally with clear, focused objectives. Be explicit about what aspect you want to improve: performance optimization, code readability, error handling, modularity, or adherence to specific design patterns. When you recognize that refinement is needed but can't articulate the specific approach (for instance, you know certain logic should be extracted into a separate function but aren't sure how), use AI to help you formulate concrete objectives before implementing changes. Describe what you're trying to achieve and ask the AI to suggest specific refactoring strategies or design patterns that would accomplish your goal, applying the same mindsets delineated in Rules 1 - 9 to help you along the way.

AI excels at identifying opportunities for refactoring and abstraction, such as recognizing repeated code that should be extracted into reusable functions or methods, and detecting poor programming patterns like deeply nested conditionals, overly long functions, tight coupling between components, sloppy or inconsistent variable naming conventions, and other poor patterns. When requesting refinements, specify the goal (e.g., "extract the data validation logic into a separate function" rather than "make this better") and verify each change against your tests (or, improving your testing as you interate to reflect the latest updates and improvements) before moving to the next improvement. This focused approach prevents the AI from making changes that, while technically sound, don't align with your project's architectural decisions. Note that AI can inadvertently break previously working code or degrade performance while making stylistic improvements. Always test thoroughly after each incremental change, and revert if the "improvement" introduces problems or doesn't provide clear benefits. 

## What separates positive from flawed examples

Flawed examples ask AI to "improve" or "clean up" code without specific objectives. The AI makes sweeping changes across multiple concerns simultaneously; renaming variables, restructuring logic, changing algorithms, adding abstractions. You can't evaluate which changes are beneficial because everything changed at once. Tests start failing but you don't know which modification caused the problem. The AI might introduce technically correct patterns that don't match your project's conventions. You waste time untangling good changes from bad ones, or worse, accept problematic changes because you can't isolate their effects.

Positive examples approach refinement systematically. You either identify specific issues yourself or ask AI to diagnose problems first, then evaluate its suggestions based on your project context. You tackle one focused objective at a time. You test after each change (improving your testing suite as you go) and revert immediately if something breaks. You recognize that not all AI suggestions are appropriate for your codebase; even good practices can be wrong if they conflict with project or field conventions for specific tendencies or methodologies. This incremental approach lets you understand each change, verify its benefit, and maintain a working codebase throughout refinement.

---

### Example 1: Vague "Improve This" Request

The user asks AI to generically improve code without specific objectives. The AI makes sweeping changes across multiple dimensions. When tests fail, the user tries to salvage the situation by listing what's wrong, but the AI's attempts to fix the problems make things worse. The conversation becomes increasingly polluted with failed attempts, conflicting constraints, and mounting confusion. What started as "improve the code" turns into a debugging nightmare where it's impossible to tell what's broken, why, or how to fix it. The user eventually gives up and has to revert everything.

````{prf:example} Vague "Improve This" Request
:label: rule10-bad-ex1
:class: dropdown

**User Prompt:**
```
This connectivity analysis code works but could be better. Clean it up and improve it.

[pastes 200 lines of working code]
```

**AI Response:**
```python
# AI makes sweeping changes:

# 1. Renamed everything (mat → connectivity_matrix, compute_stats → analyze_network)
# 2. Restructured into new class hierarchy (added AbstractConnectivityAnalyzer base)
# 3. Changed correlation algorithm (Pearson → Spearman without asking)
# 4. Added caching decorator
# 5. Introduced dependency injection
# 6. Switched from numpy to pandas for some operations
# 7. Added type hints everywhere
# 8. Extracted 5 new helper functions
# 9. Changed function signatures
# 10. Added extensive logging

# [200 lines of "improved" code that looks completely different]
```

**User Runs Tests:**
```bash
pytest tests/
# FAILED: 8 of 15 tests failing
```

**User to AI:**
```
The refactoring broke things. Problems I see:

1. Tests failing because function signatures changed
2. Spearman correlation gives different results than our validated Pearson approach
3. The new class hierarchy is too complex
4. Pandas dependency added but we're minimizing dependencies
5. The caching is causing stale results in some tests
6. Logging statements are breaking when run in test environment
7. Some extracted functions are called only once (unnecessary abstraction)

Fix these issues.
```

**AI's "Fix" Attempt:**
```python
# AI tries to address all issues simultaneously:

# 1. Keeps new function signatures but adds wrapper functions for old signatures
# 2. Adds parameter to switch between Pearson/Spearman
# 3. Keeps class hierarchy but adds backward compatibility methods
# 4. Makes pandas import optional with fallback
# 5. Adds cache invalidation logic
# 6. Wraps logging in try/except blocks
# 7. Inlines some functions but not others (inconsistent)

# NOW THERE ARE TWO VERSIONS OF EVERYTHING
# Code is even more complex than before
```

**User Tests Again:**
```bash
pytest tests/
# FAILED: 10 of 15 tests failing (WORSE!)

# New problems appeared:
# - Wrapper functions have subtle behavioral differences
# - Pearson/Spearman parameter not threaded through all calls
# - Backward compatibility methods create duplicate state
# - Optional pandas import broke type hints
# - Cache invalidation is too aggressive, now caching nothing
# - try/except around logging swallows actual errors
```

**User to AI:**
```
This is worse! Now we have:
- Wrapper functions that behave slightly differently than originals
- The Pearson/Spearman parameter isn't being passed through correctly
- More tests failing than before
- Even more complexity

Just revert the Spearman change and remove the wrappers. Keep the original 
function signatures.
```

**AI's Second "Fix":**
```python
# AI tries to revert some changes while keeping others:

# Removes wrappers but keeps new signatures (CONFLICT!)
# Changes back to Pearson but keeps the method parameter (unused now)
# Removes some of the class hierarchy but not all (inconsistent state)
# Keeps caching in some places but removes it from others
# Some logging statements removed, others kept

# THE CODE IS NOW A FRANKENSTEIN MIX OF OLD AND NEW
```

**User Tests Again:**
```bash
pytest tests/
# FAILED: 12 of 15 tests failing (EVEN WORSE!)

# The code is now in an impossible state:
# - Mix of old and new patterns
# - Inconsistent state between related functions
# - Parameters that exist but aren't used
# - Class hierarchy that's partially implemented
# - Import statements for dependencies that aren't used
```

**User's Final Attempt:**
```
I can't even tell what the code is trying to do anymore. We have:
- Vestiges of the old implementation mixed with new
- Unused parameters and imports everywhere
- Functions that call each other in confusing ways
- Tests failing for reasons I can't diagnose

This is unsalvageable. I'm reverting everything.
```

**Git Revert:**
```bash
git diff HEAD~1 HEAD | wc -l
# 847 lines changed

git revert HEAD
# Back to working code from before "improvements"
```
````

---

### Example 2: Ask AI for Diagnostic Feedback Before Refactoring

The user has working code but suspects it could be improved. Instead of asking AI to fix everything, they request analysis first. The AI identifies specific improvement opportunities. The user evaluates each suggestion in the context of their project; some are valuable, others don't fit their codebase conventions. They choose one high-priority issue to address first and give AI a focused, specific refactoring objective. This diagnostic approach works even for less experienced developers who might not recognize unideal coding patterns themselves.

````{prf:example} Ask AI for Diagnostic Feedback Before Refactoring
:label: rule10-good-ex1
:class: dropdown

**User Prompt:**
```
Review this brain connectivity analysis code and identify specific improvement 
opportunities. Don't implement anything yet; just analyze what you see.

[pastes 150 lines of working code]
```

**AI Response:**
```
I've identified several improvement opportunities:

1. **Repeated validation logic** (Lines 23, 67, 103)
   - Input shape checking repeated in 3 functions
   - NaN detection duplicated across methods
   - Recommendation: Extract into validate_connectivity_matrix() function

2. **Deeply nested conditionals** (Lines 45-62)
   - 4 levels of nesting in threshold_matrix()
   - Logic flow hard to follow
   - Recommendation: Extract conditions into guard clauses or helper functions

3. **Inconsistent naming** 
   - Mix of 'mat', 'matrix', 'conn_matrix' for same concept
   - Function names inconsistent: compute_metrics() vs calculateStats()
   - Recommendation: Standardize to 'connectivity_matrix' and snake_case throughout

4. **Magic numbers** (Lines 34, 78, 91)
   - Hardcoded values: 0.05, 100, 0.7 without explanation
   - Recommendation: Extract as named constants with documentation

5. **Tight coupling**
   - StatisticalAnalysis directly instantiates Preprocessor
   - Hard to test or swap implementations
   - Recommendation: Use dependency injection pattern

6. **Missing type hints**
   - Function signatures don't specify types
   - Recommendation: Add type annotations for better IDE support
```

**User's Thought Process:**
```
Let me evaluate each suggestion for our project:

1. Repeated validation - YES. I keep copying the same validation and it's 
   error-prone. This is a real problem.
   → Priority: HIGH

2. Nested conditionals - Hmm, the nesting actually makes the logic clear in 
   this case. The threshold logic is inherently complex.
   → Priority: SKIP (not worth changing)

3. Inconsistent naming - Good catch. This does hurt readability.
   → Priority: MEDIUM (do after validation)

4. Magic numbers - The 0.05 is standard p-value threshold, but 0.7 connectivity 
   threshold should definitely be named.
   → Priority: MEDIUM

5. Tight coupling - True, but our entire codebase doesn't use dependency 
   injection. Introducing it here would be inconsistent with project style.
   → Priority: SKIP (doesn't fit our conventions)

6. Type hints - Nice to have but not critical right now.
   → Priority: LOW (maybe later)

Decision: Start with #1 (validation extraction). Clear win, high impact.
```

**User to AI:**
```
Good analysis. Let's address #1 first: the repeated validation logic.

Create a validation function:

def validate_connectivity_matrix(matrix, allow_nan=False, require_symmetric=True):
    """
    Validate connectivity matrix meets requirements.
    
    Parameters
    ----------
    matrix : np.ndarray
        Connectivity matrix to validate
    allow_nan : bool
        Whether NaN values are acceptable
    require_symmetric : bool
        Whether matrix must be symmetric
        
    Raises
    ------
    ValueError
        If validation fails, with specific explanation of what's wrong
    """

Then refactor compute_correlation(), threshold_matrix(), and compute_metrics() 
to use this validator.

Do this incrementally:
1. Show me the validator implementation first
2. Then update compute_correlation() to use it (I'll test)
3. Then threshold_matrix() (I'll test)  
4. Then compute_metrics() (I'll test)

Don't change anything else; just the validation extraction.
```

**AI Implements Validator:**
```python
def validate_connectivity_matrix(matrix, allow_nan=False, require_symmetric=True):
    """Validate connectivity matrix meets requirements."""
    
    # Check type
    if not isinstance(matrix, np.ndarray):
        raise ValueError(f"Expected numpy array, got {type(matrix)}")
    
    # Check dimensions
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D array, got {matrix.ndim}D")
    
    # Check square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")
    
    # Check for NaN
    if not allow_nan and np.any(np.isnan(matrix)):
        raise ValueError("Matrix contains NaN values")
    
    # Check symmetry
    if require_symmetric and not np.allclose(matrix, matrix.T, equal_nan=True):
        raise ValueError("Matrix is not symmetric")
    
    return True
```

**User:**
```
Good. Now update compute_correlation() to use this validator at the start.
```

**AI Updates First Function:**
```python
def compute_correlation(timeseries):
    """Compute correlation matrix from timeseries."""
    # OLD: Manual validation code here (5 lines)
    # NEW: Use validator
    # Note: Don't validate here since input is timeseries, not connectivity matrix
    
    conn_matrix = np.corrcoef(timeseries)
    
    # Validate output
    validate_connectivity_matrix(conn_matrix, allow_nan=True, require_symmetric=True)
    
    return conn_matrix
```

**User Tests:**
```
✓ All tests pass
✓ git diff shows only compute_correlation() changed
✓ Validation logic now centralized

Continue with threshold_matrix()...
```
````

---

### Example 3: Incremental Refactoring with Testing

Following the diagnostic approach from Example 2, the user now implements one focused refactoring objective. They work incrementally: make one change, run tests, verify behavior unchanged, commit. When a change breaks something, they catch it immediately and can revert or fix because only one thing changed. This disciplined approach maintains a working codebase throughout the refactoring process. The example shows multiple small steps with verification at each point.

````{prf:example} Incremental Refactoring with Testing
:label: rule10-good-ex2
:class: dropdown

**Context:** Following Example 2's diagnosis, user is implementing validation extraction

**Step 1: Create validator function**
```python
# New function added
def validate_connectivity_matrix(matrix, allow_nan=False, require_symmetric=True):
    # [implementation from Example 2]
```

**Test Step 1:**
```bash
pytest tests/test_validation.py -v
# All validation tests pass
git add src/connectivity/validation.py
git commit -m "Add validate_connectivity_matrix function

Extracts repeated validation logic into reusable function.
No behavior changes yet; just adding the function."
```

**Step 2: Update first function**
```python
def compute_correlation(timeseries):
    """Compute correlation matrix from timeseries."""
    # Replace inline validation with validator call
    conn_matrix = np.corrcoef(timeseries)
    validate_connectivity_matrix(conn_matrix, allow_nan=True, require_symmetric=True)
    return conn_matrix
```

**Test Step 2:**
```bash
pytest tests/test_correlation.py -v
# ✓ All pass - behavior unchanged

git add src/connectivity/correlation.py
git commit -m "Update compute_correlation to use centralized validation

Removes duplicate validation code. Behavior unchanged."
```

**Step 3: Update second function**
```python
def threshold_matrix(connectivity_matrix, threshold=0.5):
    """Threshold connectivity matrix."""
    # Replace inline validation
    validate_connectivity_matrix(connectivity_matrix, 
                                allow_nan=False, 
                                require_symmetric=True)
    
    thresholded = connectivity_matrix.copy()
    thresholded[np.abs(thresholded) < threshold] = 0
    np.fill_diagonal(thresholded, 0)  # Preserve diagonal
    
    return thresholded
```

**Test Step 3:**
```bash
pytest tests/test_threshold.py -v
# ✓ All pass

git add src/connectivity/threshold.py  
git commit -m "Update threshold_matrix to use centralized validation

Second function migrated to shared validator."
```

**Step 4: Update third function**
```python
def compute_metrics(connectivity_matrix):
    """Compute graph metrics from connectivity."""
    # Replace inline validation
    validate_connectivity_matrix(connectivity_matrix,
                                allow_nan=False,
                                require_symmetric=True)
    
    # Compute metrics
    # [rest of implementation]
```

**Test Step 4:**
```bash
pytest tests/test_metrics.py -v

# FAILURE!
# test_metrics_with_nan FAILED - ValueError: Matrix contains NaN values
```

**User Analysis:**
```
Wait, the test is trying to verify that compute_metrics() handles NaN gracefully,
but now the validator is rejecting NaN before we get to that logic.

Looking at the old code... we were actually checking for NaN and handling it
specifically in this function. The validator is too strict here.

Option 1: Change validator call to allow_nan=True
Option 2: Keep the validation strict but update the test

Checking the specification... compute_metrics() should reject NaN (unlike
compute_correlation which allows it). The old test was wrong. Let me fix the test.
```

**Fix Test:**
```python
def test_metrics_with_nan():
    """compute_metrics should reject NaN values."""
    matrix = np.random.randn(10, 10)
    matrix[0, 1] = np.nan
    
    # OLD: Expected this to handle gracefully
    # NEW: Should raise ValueError
    with pytest.raises(ValueError, match="NaN"):
        compute_metrics(matrix)
```

**Test Again:**
```bash
pytest tests/test_metrics.py -v
# ✓ All pass now

git add src/connectivity/metrics.py tests/test_metrics.py
git commit -m "Update compute_metrics to use centralized validation

Fixed test to match intended behavior: NaN should be rejected.
All validation logic now centralized."
```

**Final Verification:**
```bash
# Run full test suite
pytest tests/ -v
# ✓ All 47 tests pass

# Check what changed overall
git log --oneline -5
# 4 commits, each focused on one change
# Easy to review, easy to revert if needed
```

**Result:**
- Validation logic successfully centralized
- All three functions refactored
- One test bug caught and fixed
- Code works throughout entire process
- Clear commit history documents what happened
````

---

### Example 4: Performance Optimization with Baseline Metrics

The user identifies a performance bottleneck through profiling. Before optimizing, they establish baseline metrics. They request one specific optimization from AI, verify it actually improves performance, and confirm correctness is maintained. When an "optimization" actually makes things worse, they catch it immediately by comparing to the baseline and revert. This metrics-driven approach prevents premature optimization and ensures changes provide real benefits.

````{prf:example} Performance Optimization with Baseline Metrics
:label: rule10-good-ex3
:class: dropdown

**User Analysis:**
```
Current performance profiling shows:
- Total processing: 45 seconds per subject
- Breakdown:
  * Loading data: 3s
  * Motion correction: 8s
  * Correlation computation: 28s  ← BOTTLENECK
  * Statistical tests: 6s

The correlation computation is the problem. Let me look at the code...
```

**Current Implementation:**
```python
def compute_connectivity(timeseries):
    """
    Compute pairwise correlations.
    
    Args:
        timeseries: (n_regions, n_timepoints) array
    """
    n_regions = timeseries.shape[0]
    connectivity = np.zeros((n_regions, n_regions))
    
    # Nested loops - probably the bottleneck
    for i in range(n_regions):
        for j in range(n_regions):
            connectivity[i, j] = np.corrcoef(timeseries[i], timeseries[j])[0, 1]
    
    return connectivity
```

**Establish Baseline:**
```python
# Create benchmark
import time

def benchmark_connectivity():
    """Benchmark current implementation."""
    timeseries = np.random.randn(100, 500)  # Typical data size
    
    times = []
    for _ in range(10):
        start = time.time()
        result = compute_connectivity(timeseries)
        times.append(time.time() - start)
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

baseline = benchmark_connectivity()
print(f"Baseline: {baseline['mean']:.3f}s ± {baseline['std']:.3f}s")
# Baseline: 2.847s ± 0.043s
```

**User to AI:**
```
Optimize compute_connectivity() for speed. Current performance: 2.85 seconds.

SPECIFIC OPTIMIZATION:
Replace the nested loop with vectorized numpy operation. The current 
implementation calls np.corrcoef() 10,000 times (100x100). We should call 
it once on the entire timeseries matrix.

CONSTRAINTS:
- Must maintain identical output (within floating point tolerance)
- Must handle NaN values the same way
- Should use numpy only (no new dependencies)

TARGET: Under 0.1 seconds (30x improvement)

Show implementation, and I'll benchmark it.
```

**AI Optimized Version:**
```python
def compute_connectivity(timeseries):
    """
    Compute pairwise correlations using vectorized operations.
    
    Args:
        timeseries: (n_regions, n_timepoints) array
    """
    # Single call to corrcoef instead of nested loops
    connectivity = np.corrcoef(timeseries)
    return connectivity
```

**User Benchmarks:**
```python
# Test correctness first
old_result = compute_connectivity_old(test_data)
new_result = compute_connectivity(test_data)
assert np.allclose(old_result, new_result)
print("✓ Correctness verified")

# Benchmark new version
new_perf = benchmark_connectivity()
print(f"New: {new_perf['mean']:.3f}s ± {new_perf['std']:.3f}s")
# New: 0.008s ± 0.001s

# Compare
speedup = baseline['mean'] / new_perf['mean']
print(f"Speedup: {speedup:.1f}x")
# Speedup: 355.9x

print("✓ Target achieved (>30x improvement)")
```

**Full Test Suite:**
```bash
pytest tests/test_connectivity.py -v
# ✓ All 15 tests pass
# No behavior changes despite major performance improvement
```

**User:**
```
Excellent. Commit this optimization.
```

**Alternative Scenario: "Optimization" Makes Things Worse:**

**AI "Optimized" Version:**
```python
def compute_connectivity_v2(timeseries):
    """Optimize using sparse matrix operations."""
    from scipy.sparse import csr_matrix
    
    # Convert to sparse (thinking this will be faster)
    sparse_ts = csr_matrix(timeseries)
    # [complicated sparse correlation implementation]
```

**User Benchmarks:**
```python
new_perf = benchmark_connectivity_v2()
print(f"Sparse version: {new_perf['mean']:.3f}s")
# Sparse version: 4.231s

print(f"Comparison to baseline: {new_perf['mean'] / baseline['mean']:.2f}x")
# Comparison to baseline: 1.49x SLOWER

print("✗ This 'optimization' actually made it worse!")
```

**User Summary:**
```
This sparse matrix approach is actually slower than the original. The overhead 
of sparse operations doesn't pay off for dense correlation matrices.

Reverting to the simple vectorized version which was 355x faster.
```
````

### Example 5: AI Breaks Code During Refactoring

The user requests a specific refactoring. The AI implements it but subtly changes behavior in the process. Because the user tests after each change (following the incremental approach), they catch the breakage immediately. They can identify exactly what went wrong, provide corrected requirements, and try again. If they'd made multiple changes at once, they wouldn't know which change caused the problem.

````{prf:example} AI Breaks Code During Refactoring
:label: rule10-good-ex4
:class: dropdown

**User Request:**
```
Extract the network thresholding logic from analyze_network() into a separate 
function threshold_by_density().

Current code:
```
```python
def analyze_network(connectivity_matrix, density=0.2):
    """Analyze network properties at given density."""
    # Threshold to keep top X% of connections
    threshold_value = np.percentile(np.abs(connectivity_matrix), 
                                   100 * (1 - density))
    
    thresholded = connectivity_matrix.copy()
    thresholded[np.abs(thresholded) < threshold_value] = 0
    
    # Compute metrics on thresholded network
    metrics = compute_graph_metrics(thresholded)
    return metrics
```

**User to AI:**
```
Extract the thresholding logic (lines 3-6) into:

def threshold_by_density(connectivity_matrix, density=0.2):
    """Keep only top density% of connections by absolute value."""
    # Extract threshold computation and application here
    return thresholded_matrix

Then update analyze_network() to call this function.
```

**AI Implementation:**
```python
def threshold_by_density(connectivity_matrix, density=0.2):
    """Keep only top density% of connections by absolute value."""
    threshold_value = np.percentile(connectivity_matrix,  # BUG: removed np.abs()
                                   100 * (1 - density))
    
    thresholded = connectivity_matrix.copy()
    thresholded[connectivity_matrix < threshold_value] = 0  # BUG: wrong comparison
    
    return thresholded

def analyze_network(connectivity_matrix, density=0.2):
    """Analyze network properties at given density."""
    thresholded = threshold_by_density(connectivity_matrix, density)
    metrics = compute_graph_metrics(thresholded)
    return metrics
```

**User Tests:**
```bash
pytest tests/test_network_analysis.py -v

FAILED tests/test_network_analysis.py::test_threshold_by_density
FAILED tests/test_network_analysis.py::test_analyze_symmetric_network

# Two tests failed after the refactoring
```

**User Investigation:**
```python
# Check what's wrong
test_matrix = np.array([
    [ 1.0,  0.8, -0.6,  0.3],
    [ 0.8,  1.0,  0.2, -0.7],
    [-0.6,  0.2,  1.0,  0.4],
    [ 0.3, -0.7,  0.4,  1.0]
])

result = threshold_by_density(test_matrix, density=0.5)

print("Result:")
print(result)
# Result shows negative values are kept, positive weak connections removed
# This is BACKWARDS from expected behavior

print("\nExpected: Keep strongest connections by absolute value")
print("Actual: Threshold is based on raw values, not absolute values")

# The bug: AI removed np.abs() from percentile calculation
# AND used wrong comparison (< instead of checking absolute value)
```

**User to AI:**
```
The refactoring introduced two bugs:

BUG 1: Line 3 should be np.percentile(np.abs(connectivity_matrix), ...)
       You removed the np.abs(), so threshold is computed on raw values
       instead of absolute values. This means strong negative correlations
       get incorrectly removed.

BUG 2: Line 6 should threshold by absolute value: 
       thresholded[np.abs(thresholded) < threshold_value] = 0
       Your version only removes values below threshold, not weak values
       in general.

The function should keep the top X% strongest connections regardless of sign.

Fix both bugs.
```

**AI Corrected Version:**
```python
def threshold_by_density(connectivity_matrix, density=0.2):
    """Keep only top density% of connections by absolute value."""
    # Threshold based on absolute values
    threshold_value = np.percentile(np.abs(connectivity_matrix),
                                   100 * (1 - density))
    
    thresholded = connectivity_matrix.copy()
    # Zero out connections weaker than threshold (by absolute value)
    thresholded[np.abs(thresholded) < threshold_value] = 0
    
    return thresholded
```

**User Tests Again:**
```bash
pytest tests/test_network_analysis.py -v
# ✓ All tests pass

# Verify behavior explicitly
result = threshold_by_density(test_matrix, density=0.5)
strong_connections = np.sum(result != 0) - 4  # Exclude diagonal
print(f"Strong connections kept: {strong_connections}")
# Strong connections kept: 4 (top 50% of 8 off-diagonal connections)

print("✓ Behavior now matches original implementation")
```

**User Reflection:**
```
Good thing I tested immediately after the refactoring. If I'd made multiple 
changes at once, I wouldn't know which one introduced the bug.

The incremental approach caught this quickly:
1. One change (extraction)
2. Test immediately  
3. Caught bug
4. Fixed and verified
5. Now safe to move to next refactoring

If I'd asked AI to "improve the whole module," this bug would be buried 
among dozens of changes and much harder to diagnose.
```
````