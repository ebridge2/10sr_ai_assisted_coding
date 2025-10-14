(rules:rule_9)=
# Rule 9: Critically Review Generated Code

Be skeptical about AI's claims of success; the models tend to claim success even when they haven't really solved the problem. You always need to test the solution independently. Read and understand the code to ensure it solves problems in ways that make sense for your domain and match your prior expectation of how the problem should be solved (e.g., how you anticipated a solution looking based on your pseudocode or architecture schematics you developed in Rule 4). AI-generated code requires careful human review to ensure scientific appropriateness, methodological soundness, and alignment with domain standards.

## What separates positive from flawed examples

Flawed examples accept AI-generated code at face value because it runs without errors and the AI claims it's correct. You don't actually read through the implementation to verify it does what you think. The code might be using inappropriate algorithms, violating domain conventions, or producing subtly wrong results. You discover the problems later, possibly after publishing analyses based on the flawed code.

Positive examples involve thorough critical review of generated code. You read the implementation and verify it matches your conceptual understanding of the solution. You check that algorithms are scientifically appropriate for your domain. You verify outputs make sense and align with established methods. You catch issues before the code gets used for real analyses.

---

## Example 1: Accepting Code Without Review

The user asks for an implementation, the AI provides code that runs without errors, and the user accepts it because the AI says it's correct. The user never actually reads the implementation to understand what it's doing. Later, the code produces scientifically invalid results because it used an inappropriate method. The problem could have been caught with basic code review.

````{prf:example} Trusting AI claims without verification
:label: rule10-bad-ex1
:class: dropdown

**User Prompt:**
```
Implement false discovery rate correction for multiple comparisons in 
neuroimaging analysis.
```

**AI Response:**
```python
def fdr_correction(p_values, alpha=0.05):
    """Apply FDR correction to p-values."""
    from scipy.stats import false_discovery_control
    return false_discovery_control(p_values, alpha=alpha)

# This implementation is correct and follows the Benjamini-Hochberg procedure.
# It will properly control the false discovery rate at your specified alpha level.
```

**User:**
```
Great, thanks! [Uses code in analysis without further review]
```

**Problem (discovered months later):**
```
Reviewer: "Your FDR correction doesn't account for spatial dependencies 
in neuroimaging data. You should be using FWE correction or cluster-based 
thresholding, not standard FDR. The Benjamini-Hochberg procedure assumes 
independence, which is violated in brain imaging."

The AI gave you working code for FDR correction, but it wasn't the right 
method for neuroimaging. A domain expert would have caught this immediately 
upon reading the code, but you never reviewed whether the approach was 
scientifically appropriate for your use case.
```
````

---

## Example 2: Critical Review Catches Methodological Issues

The user receives code from AI and immediately reviews it critically. They notice the implementation doesn't match standard practice in their field. They ask clarifying questions and verify the approach is scientifically appropriate. They catch a methodological issue before using the code for real analysis.

````{prf:example} Reading code to verify scientific appropriateness
:label: rule10-good-ex1
:class: dropdown

**User Prompt:**
```
Implement statistical comparison of connectivity matrices between two groups 
(patients vs controls) for neuroimaging analysis. We have connectivity 
matrices from 30 patients and 30 controls.
```

**AI Response:**
```python
def compare_connectivity(group1_matrices, group2_matrices):
    """
    Compare connectivity between two groups using t-tests.
    
    Parameters
    ----------
    group1_matrices : list of np.ndarray
        Connectivity matrices for group 1 (n_subjects, n_regions, n_regions)
    group2_matrices : list of np.ndarray
        Connectivity matrices for group 2
        
    Returns
    -------
    t_stats : np.ndarray
        T-statistics for each connection
    p_values : np.ndarray
        P-values for each connection
    """
    from scipy.stats import ttest_ind
    
    # Stack matrices
    g1 = np.stack(group1_matrices)  # (n_subj, n_reg, n_reg)
    g2 = np.stack(group2_matrices)
    
    # T-test for each connection
    n_regions = g1.shape[1]
    t_stats = np.zeros((n_regions, n_regions))
    p_values = np.zeros((n_regions, n_regions))
    
    for i in range(n_regions):
        for j in range(n_regions):
            t_stats[i, j], p_values[i, j] = ttest_ind(g1[:, i, j], g2[:, i, j])
    
    return t_stats, p_values

# This performs element-wise t-tests comparing each connection between groups.
```

**User Review:**
```
Wait, let me read through this...

ISSUES I SEE:
1. This does element-wise t-tests on every connection. With 100 regions, 
   that's 10,000 comparisons. No multiple comparison correction mentioned.

2. It includes the diagonal (self-connections = 1.0 for everyone). These 
   don't vary and shouldn't be tested.

3. It tests both i,j and j,i separately, but connectivity matrices are 
   symmetric. This doubles the comparisons unnecessarily.

4. More fundamentally: is element-wise testing even appropriate? In 
   neuroimaging, we often use permutation testing or network-based 
   statistics to account for dependencies.

Let me ask for clarification...
```

**User to AI:**
```
I see several issues with this approach:

1. You're doing 10,000 t-tests (100x100 matrix) with no multiple comparison 
   correction. That's guaranteed to give false positives.

2. You're testing the diagonal, which is always 1.0 for everyone.

3. You're testing both (i,j) and (j,i) separately when matrices are symmetric.

4. Most importantly: element-wise testing ignores the network structure. 
   Standard practice in neuroimaging is either:
   - Network-based statistics (Zalesky et al. 2010)
   - Permutation testing with max-statistic
   - Graph-level metrics instead of edge-level

Revise the approach to:
- Use only upper triangle (exclude diagonal)
- Apply FWE correction via permutation testing
- Or better: compare graph metrics (clustering, efficiency) between groups

Which approach is most appropriate depends on our hypothesis. We're looking 
for overall network differences, not specific connections, so graph metrics 
make more sense.

Implement comparison of graph-level metrics (global efficiency, clustering 
coefficient, modularity) with permutation testing.
```
````

---

## Example 3: Verifying Implementation Matches Expectations

The user has a clear expectation of how the algorithm should work (from Rule 4). When reviewing the generated code, they notice the implementation doesn't match their understanding. They identify the discrepancy and get it corrected before using the code.

````{prf:example} Checking implementation matches prior expectations
:label: rule10-good-ex2
:class: dropdown

**User Context (from Rule 4 planning):**
```
I need hierarchical clustering of brain connectivity patterns.

Expected approach:
1. Compute distance matrix (1 - correlation)
2. Apply agglomerative clustering with Ward linkage
3. Determine optimal number of clusters using silhouette score
4. Return cluster assignments and dendrogram

Ward linkage minimizes within-cluster variance, which is standard for 
neuroimaging because we want compact, similar clusters.
```

**AI Response:**
```python
def cluster_connectivity(connectivity_matrices):
    """Hierarchical clustering of connectivity patterns."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    
    # Flatten matrices to vectors
    n_subjects = len(connectivity_matrices)
    n_regions = connectivity_matrices[0].shape[0]
    vectors = np.array([mat.flatten() for mat in connectivity_matrices])
    
    # Compute distances and cluster
    distances = pdist(vectors, metric='euclidean')
    linkage_matrix = linkage(distances, method='complete')
    
    # Use fixed number of clusters
    clusters = fcluster(linkage_matrix, t=3, criterion='maxclust')
    
    return clusters, linkage_matrix
```

**User Review:**
```
Hold on, this doesn't match my expectations:

ISSUES:
1. Uses 'complete' linkage instead of 'ward'. Complete linkage tends to 
   create elongated clusters, not the compact clusters I want.

2. Uses 'euclidean' distance on flattened matrices. I specified using 
   (1 - correlation) as the distance metric, which is standard for 
   connectivity patterns.

3. Uses fixed 3 clusters instead of determining optimal k using 
   silhouette score as I specified.

4. Returns cluster assignments but I also wanted the dendrogram for 
   visualization.

Let me verify my understanding is correct by checking the literature...
[Checks standard neuroimaging clustering papers]

Yes, Ward linkage with correlation-based distance is standard. This 
implementation would give wrong results. Let me correct it.
```

**User to AI:**
```
This implementation doesn't match the standard approach for connectivity 
clustering. Please revise:

REQUIRED CHANGES:
1. Distance metric: Use (1 - correlation), not Euclidean
   - Connectivity patterns are best compared by correlation
   - Euclidean distance on flattened matrices doesn't make sense here

2. Linkage method: Use 'ward', not 'complete'
   - Ward minimizes within-cluster variance (standard for neuroimaging)
   - Complete linkage creates elongated clusters (not what we want)

3. Optimal k selection: Don't hardcode 3 clusters
   - Test k from 2 to 10
   - Select k that maximizes mean silhouette score
   - Return both the optimal k and scores for all k tested

4. Return dendrogram: Include dendrogram data structure for plotting

Implement these corrections based on standard hierarchical clustering 
practice in neuroimaging (see Yeo et al. 2011 for reference approach).
```
````

---

## Example 4: Testing Against Known Results

The user generates code and immediately tests it against cases where they know the correct answer. This catches implementation errors that might not be obvious from reading the code alone. The test reveals the implementation is computing the wrong thing despite looking reasonable.

````{prf:example} Validating against known ground truth
:label: rule10-good-ex3
:class: dropdown

**AI Generated Code:**
```python
def compute_graph_efficiency(adjacency_matrix):
    """
    Compute global efficiency of brain network.
    
    Global efficiency measures how efficiently information can travel 
    through the network.
    """
    import networkx as nx
    
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Compute efficiency
    efficiency = nx.global_efficiency(G)
    
    return efficiency
```

**User Review:**
```
The code looks reasonable, but let me test it against a case where I know 
the correct answer...
```

**User Test:**
```python
# Test case 1: Complete graph (all nodes connected)
# Expected efficiency: 1.0 (shortest path between any two nodes is 1)
complete = np.ones((5, 5))
np.fill_diagonal(complete, 0)
eff_complete = compute_graph_efficiency(complete)
print(f"Complete graph efficiency: {eff_complete}")
# Expected: 1.0

# Test case 2: Path graph (nodes in a line: 0-1-2-3-4)
# Expected efficiency: 0.4 (computed by hand)
path = np.array([
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
])
eff_path = compute_graph_efficiency(path)
print(f"Path graph efficiency: {eff_path}")
# Expected: 0.4

# Test case 3: Disconnected graph
# Expected efficiency: < 1.0 (infinite distances should reduce efficiency)
disconnected = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])
eff_disconnected = compute_graph_efficiency(disconnected)
print(f"Disconnected graph efficiency: {eff_disconnected}")
# Expected: ~0.33
```

**Results:**
```
Complete graph efficiency: 1.0  ✓ Correct
Path graph efficiency: 0.6     ✗ Wrong! Expected 0.4
Disconnected graph efficiency: 0.5  ✗ Wrong! Expected ~0.33
```

**User:**
```
The results don't match my hand calculations. Let me check the networkx 
documentation...

OH! The issue is that networkx.from_numpy_array treats the adjacency matrix 
as WEIGHTED by default. My test matrices are binary (0/1), but networkx is 
interpreting edge weights.

For brain networks, we typically want:
- Binary: unweighted connectivity (connected or not)
- Weighted: connection strength (correlation values)

The current implementation doesn't specify which, and the default behavior 
is wrong for my binary test cases.

Let me revise to explicitly handle both weighted and unweighted cases.
```

**User to AI:**
```
The implementation has an issue: it doesn't specify whether the network is 
weighted or unweighted, and networkx defaults to weighted interpretation.

Revise to:
1. Add a 'weighted' parameter (default=False for binary connectivity)
2. For weighted=False: convert adjacency to binary before computing efficiency
3. For weighted=True: use edge weights in efficiency calculation
4. Document the difference clearly in the docstring

Test both cases to ensure correct results:
- Binary complete graph should give exactly 1.0
- Binary path graph should give exactly 0.4
- Weighted networks should use edge weights in distance calculation
```
````
