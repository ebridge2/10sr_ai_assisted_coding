(rules:rule_2)=
# Rule 2: Distinguish Problem Framing from Coding

Framing a problem in a programmatic way and coding are not the same thing {cite}`martin2008clean`. Programmatic problem framing is problem solvinging is problem-solving: understanding the domain, decomposing complex problems, finding the right levels of abstraction, designing algorithms, and making architectural decisions. Coding is the mechanical translation of these concepts into executable syntax in a programming language. Using AI coding tools effectively requires that you deeply understand the problem from a programmatic perspective that you are trying to solve; in most cases this understanding transcends the particular programming language, and the actual code implementation itself. AI tools excel at coding tasks, generating syntactically correct implementations from well-specified requirements, but they currently require human guidance for programmatic problem framing decisions that involve domain expertise, methodological choices, and scientific reasoning. You can't effectively guide or review what you don't understand, so establish fluency in at least one programming language and fundamental concepts before leveraging AI assistance. This foundation allows you to spot when generated code deviates from best practices or introduces subtle bugs. Without this knowledge, you're essentially flying blind, unable to distinguish between elegant solutions and convoluted workarounds.

## What separates positive from flawed examples

Flawed examples make vague requests without understanding the problem structure. You get code that might run, but you have no idea if it's solving the problem correctly, efficiently, or in a way that makes scientific sense. You can't debug it when it breaks, can't explain what it's doing, and can't verify its correctness.

Positive examples demonstrate clear understanding of the problem at a conceptual level. You specify inputs, outputs, constraints, and expected behavior. You can articulate what success looks like and why. You provide enough detail that you could pseudocode the solution yourself, even if the syntax would be messy. This gives the AI a clear target and gives you the ability to evaluate whether what comes back is reasonable.

---

## Example 1: Vibe Coding Without Understanding

The user has no idea what problem they're actually trying to solve. Can't tell if correlation is even the right measure (could need partial correlation, mutual information, something else entirely). There's no specification of what "analyze" means scientifically. Can't evaluate if 0.7 is a meaningful threshold for anything. The loop implementation is inefficient, but the user has no way to know that. The code includes the diagonal (self-correlation equals 1) in the statistics, which is almost certainly wrong. The user can't debug this when it inevitably breaks and can't explain what it's doing to collaborators.

````{prf:example} Vague request without problem comprehension
:label: rule2-bad-ex1
:class: dropdown

**User Prompt:**
```
I need to analyze brain connectivity data. Make it work with my files.
```

**AI Response:**
```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def analyze_connectivity(file_path):
    """Analyze brain connectivity from file."""
    # Load data
    data = pd.read_csv(file_path)
    
    # Calculate connectivity
    connectivity = np.zeros((len(data.columns), len(data.columns)))
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            connectivity[i, j] = pearsonr(data.iloc[:, i], data.iloc[:, j])[0]
    
    # Analyze
    avg_connectivity = np.mean(connectivity)
    strong_connections = np.sum(connectivity > 0.7)
    
    return {
        'average': avg_connectivity,
        'strong_connections': strong_connections,
        'matrix': connectivity
    }
```
````

---

## Example 2: Problem-First Specification

The user clearly articulates the problem structure: inputs, outputs, constraints. They specify what "connectivity" means in this context (Pearson correlation, not something else). They provide concrete requirements the AI can actually implement. They list validation criteria so they can verify correctness themselves. The NaN handling requirement gets explained explicitly. Now the user can actually check if this solution makes sense. They can debug issues because they understand what should happen. They can explain to collaborators what this code does and why it does it that way.

````{prf:example} Clear problem decomposition before implementation
:label: rule2-good-ex1
:class: dropdown

**User Prompt:**
```
I need to compute functional connectivity between brain regions from fMRI timeseries data. Here's what I understand about the problem:

**Problem structure:**
- Input: NxT array where N=number of regions, T=number of timepoints
- Each row is the BOLD signal timeseries for one region
- Need pairwise correlation between all region pairs
- Output: NxN symmetric correlation matrix

**Key considerations:**
- Some timeseries may have NaN values (motion-censored timepoints)
- Should use Pearson correlation (standard in field)
- Need to handle the NaN case with pairwise-complete observations
- Diagonal should be 1.0 (region correlated with itself)
- Matrix should be symmetric

**What I want:**
Create function compute_functional_connectivity(timeseries, method='pearson') that:
- Takes (n_regions, n_timepoints) array
- Returns (n_regions, n_regions) correlation matrix
- Uses numpy/scipy efficiently (no explicit loops)
- Handles NaN via pairwise deletion
- Includes input validation (correct shape, at least 2 timepoints)

I can verify correctness by checking:
- Output shape matches (n_regions, n_regions)
- Diagonal is all 1.0
- Matrix is symmetric
- Values in [-1, 1]
- Matches manual calculation on small test case
```

**AI Response:**
```python
import numpy as np
from numpy.ma import corrcoef

def compute_functional_connectivity(timeseries, method='pearson'):
    """
    Compute functional connectivity from fMRI timeseries.
    
    Parameters
    ----------
    timeseries : ndarray, shape (n_regions, n_timepoints)
        BOLD signal timeseries for each region
    method : str
        Correlation method ('pearson' only currently supported)
        
    Returns
    -------
    connectivity : ndarray, shape (n_regions, n_regions)
        Functional connectivity matrix
        
    Raises
    ------
    ValueError
        If timeseries has wrong shape or insufficient timepoints
    """
    if timeseries.ndim != 2:
        raise ValueError(f"Expected 2D array, got {timeseries.ndim}D")
    
    n_regions, n_timepoints = timeseries.shape
    if n_timepoints < 2:
        raise ValueError(f"Need at least 2 timepoints, got {n_timepoints}")
    
    if method != 'pearson':
        raise NotImplementedError(f"Method '{method}' not supported")
    
    # Use masked array to handle NaN with pairwise deletion
    masked_data = np.ma.masked_invalid(timeseries)
    connectivity = np.ma.corrcoef(masked_data)
    
    # Convert masked array to regular array, filling any remaining NaN with 0
    connectivity = np.ma.filled(connectivity, fill_value=0.0)
    
    return connectivity
```
````

---

## Example 3: Algorithmic Understanding Guides Implementation

The user understands the algorithm at a conceptual level before asking for any code. They specify the modularity formula explicitly rather than hoping the AI gets it right. They provide concrete stopping criteria and expected behavior. They can verify each component works correctly (modularity calculation, gain computation). They know what "reasonable" output looks like for their domain. They can debug by checking intermediate modularity values match what they expect. They have a comparison point (networkx implementation) for validation.

````{prf:example} Specifying the algorithm, not just the goal
:label: rule2-good-ex2
:class: dropdown

**User Prompt:**
```
I need to implement the Louvain algorithm for community detection in brain networks. I understand the algorithm conceptually:

**Algorithm structure:**
1. Start with each node in its own community
2. Iteratively move nodes to maximize modularity gain
3. Aggregate communities into super-nodes
4. Repeat until no modularity improvement

**What I need implemented:**
- Function to compute modularity Q given partition
- Function to compute modularity gain from moving node i to community c
- Main loop that performs the optimization

**Key details I understand:**
- Modularity formula: Q = (1/2m) Σ[Aij - kikj/2m]δ(ci,cj)
- Need to handle weighted networks (edge weights from tractography)
- Should use Newman's fast modularity gain calculation
- Typical stopping criterion: ΔQ < 0.0001

**Expected behavior:**
- Input: (N,N) weighted adjacency matrix
- Output: community assignment vector, final modularity value
- Should converge in < 100 iterations for typical brain networks

I can validate by:
- Checking Q increases at each step
- Testing on known community structures (e.g., synthetic modular networks)
- Comparing results to networkx.community.louvain_communities
```

**AI Response:**
```python
import numpy as np
from collections import defaultdict

def compute_modularity(adjacency, communities):
    """Compute modularity Q for given community partition."""
    n = len(adjacency)
    m = np.sum(adjacency) / 2  # Total edge weight
    
    if m == 0:
        return 0.0
    
    Q = 0.0
    for i in range(n):
        for j in range(n):
            if communities[i] == communities[j]:
                k_i = np.sum(adjacency[i, :])
                k_j = np.sum(adjacency[:, j])
                Q += adjacency[i, j] - (k_i * k_j) / (2 * m)
    
    return Q / (2 * m)

def louvain_communities(adjacency, resolution=1.0, max_iter=100, tol=1e-4):
    """
    Detect communities using Louvain algorithm.
    
    Parameters
    ----------
    adjacency : ndarray, shape (n, n)
        Weighted adjacency matrix
    resolution : float
        Resolution parameter (default=1.0)
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance for modularity change
        
    Returns
    -------
    communities : ndarray
        Community assignment for each node
    modularity : float
        Final modularity value
    """
    # [Implementation follows specification...]
    # Returns communities, final_Q
```
````

## References

```{bibliography}
:filter: docname in docnames
```