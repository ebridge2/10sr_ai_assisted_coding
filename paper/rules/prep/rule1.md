(rules:rule_1)=
# Rule 1: Gather Domain Knowledge Before Implementation

Know your problem space before writing code. Understand data shapes, missing data patterns, field-specific libraries, and existing implementations that could serve as models. You don't need to be an expert initially; use AI to help research domain standards, available datasets, common approaches, and implementation patterns before diving into coding. This reconnaissance phase prevents you from reinventing wheels or violating field conventions. Share your current understanding level with the AI and iteratively build context through targeted questions about tools, data structures, and best practices, asking for specific references and paper summaries. This upfront investment ensures that your code aligns with community standards and handles real-world data appropriately.

## What separates positive from flawed examples

Flawed examples jump straight to implementation without understanding the domain. You end up with generic code that ignores field-specific conventions, misses critical preprocessing steps, and uses inappropriate methods. Worse, you can't even tell if what you got back makes sense because you don't know what the right approach looks like.

Positive examples start by explicitly stating what you do and don't know, and systematically build understanding through targeted questions. You ask for specific references, methodological papers, standard tools, and data formats. You iteratively refine your understanding before writing a single line of code. This way, when you finally ask for implementation, you can actually evaluate what the AI produces.

---

## Example 1: Insufficient Domain Research

Jumping to implementation without domain understanding leaves you with code you can't evaluate. You don't know if brain networks should be weighted or binary. Critical preprocessing steps like thresholding strategies and normalization approaches are completely missing. The code uses metrics without domain justification (average path length might be undefined for disconnected graphs, which are common in brain networks). Field-standard tools like Brain Connectivity Toolbox get ignored entirely. You have no way to verify if this approach aligns with neuroscience conventions, and you're stuck with code you fundamentally can't assess.

````{prf:example} Vague request without domain context
:label: rule1-bad-ex1
:class: dropdown

**User Prompt:**
```
I need to implement graph theoretical analysis for brain networks. 
Can you write me a function?
```

**AI Response:**
```python
import networkx as nx
import numpy as np

def analyze_brain_network(adjacency_matrix):
    """Analyze a brain network using graph theory."""
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Calculate basic metrics
    degree = dict(G.degree())
    clustering = nx.clustering(G)
    path_length = nx.average_shortest_path_length(G)
    
    return {
        'degree': degree,
        'clustering': clustering,
        'path_length': path_length
    }
```
````

---

## Example 2: Thorough Domain Exploration

Starting with explicit acknowledgment of your knowledge level ("understand basic graph theory but new to brain connectivity") sets up productive learning. Asking for specific references before any coding gets you the foundational papers you need. Building understanding iteratively through targeted questions teaches you about standard tools (bctpy), typical data formats, and methodological considerations. Getting concrete papers lets you verify information independently. Most importantly, confirming understanding before moving to implementation means you're now equipped to evaluate whether the AI's eventual code actually makes sense.

````{prf:example} Iterative domain knowledge gathering
:label: rule1-good-ex1
:class: dropdown

**User Prompt (Initial):**
```
I need to implement graph theoretical analysis for brain networks from diffusion MRI tractography. I understand basic graph theory but I'm new to applying it to brain connectivity. What should I know before implementing this? Please provide key references I should review.
```

**AI Response:**
```
For brain network analysis from tractography, you should understand:

1. **Data representation**: 
   - Nodes are brain regions (from parcellations)
   - Edges are white matter connections (streamline counts or FA values)
   - Networks are typically weighted and undirected

2. **Key methodological papers**:
   - Bullmore & Sporns (2009) Nature Reviews Neuroscience - foundational review
   - Rubinov & Sporns (2010) NeuroImage - Brain Connectivity Toolbox paper
   - Fornito et al. (2013) NeuroImage - interpretation of graph metrics

3. **Standard Python libraries**:
   - networkx: general graph analysis
   - bctpy: Python port of Brain Connectivity Toolbox
   - nilearn.connectome: neuroimaging-specific tools

4. **Critical preprocessing considerations**:
   - Threshold selection (Roberts et al. 2017 NeuroImage)
   - Edge weight handling (Betzel et al. 2019 Network Neuroscience)
   - Log-transformation for streamline counts (Civier et al. 2019)

Would you like me to summarize specific papers or discuss data formats?
```

**User Follow-up:**
```
Can you summarize the main points from Rubinov & Sporns 2010 about which metrics are most reliable? Also, what does typical tractography connectivity data look like?
```

**AI Response:**
```
From Rubinov & Sporns 2010, the most reliable metrics are:
- Degree distribution, clustering coefficient, path length (robust)
- Modularity (sensitive to resolution parameter - use carefully)
- Small-worldness (requires proper null model selection)

Typical tractography data structure:
- NxN connectivity matrix where N = number of brain regions (68-1000 typical)
- Edge weights = streamline counts or mean FA along tracts
- Often sparse (60-80% of possible connections absent)
- Symmetric for undirected tractography

Standard formats:
- .mat files with 'connectivity', 'region_labels' variables
- CSV files with region pairs and weights
- Common parcellations: Schaefer (100-1000 ROIs), Desikan-Killiany (68 ROIs)

Benchmark datasets:
- Human Connectome Project (preprocessed matrices available)
- MICCAI tractography challenges

Ready to proceed with implementation?
```

**User:**
```
Yes, I've reviewed the BCT documentation and Fornito et al. Now I'm ready to implement. Here's what I need...
```
````

---

## Example 3: Learning Community Standards

Asking about standards and existing implementations before writing any code gets you oriented in what the field actually uses. You learn that MNE-Python is the neuroscience standard, not just a general-purpose tool. You understand preprocessing requirements specific to EEG (highpass filtering, artifact rejection, reference choice). You get reporting conventions needed for publications (frequency band definitions, dB scale, normalization approaches). Now you have concrete documentation to reference and can make informed decisions about using existing tools versus implementing custom solutions.

````{prf:example} Discovering field conventions before implementation
:label: rule1-good-ex2
:class: dropdown

**User Prompt:**
```
I'm analyzing EEG data and need to compute power spectral density. What are the standard approaches in neuroscience? Are there existing implementations I should look at before writing my own?
```

**AI Response:**
```
Standard approaches for EEG power spectral density:

1. **Methods** (from Cohen 2014, "Analyzing Neural Time Series Data"):
   - Welch's method (most common - reduces variance via averaging)
   - Multitaper method (better for short segments)
   - Wavelet transform (for time-frequency analysis)

2. **Standard Python implementations**:
   - MNE-Python: mne.time_frequency.psd_welch() - neuroscience standard
   - SciPy: scipy.signal.welch() - general purpose
   - FOOOF: parameterizes 1/f and oscillatory components

3. **Key preprocessing steps** (Jas et al. 2018 MNE paper):
   - Highpass filter ≥0.1 Hz to remove drift
   - Artifact rejection (ICA or rejection by amplitude)
   - Reference choice matters (average, mastoid, REST)

4. **Reporting conventions**:
   - Frequency bands: delta 1-4Hz, theta 4-8Hz, alpha 8-13Hz, etc.
   - Use dB scale (10*log10(power)) for visualization
   - Normalize by total power or baseline period

Example implementations: MNE tutorials at https://mne.tools/stable/auto_tutorials

Would you like code following MNE conventions?
```
````

---

**Key Takeaways:**
- Always research domain conventions before writing code
- Use AI to identify relevant papers, standard tools, best practices
- Build understanding iteratively through targeted questions
- Verify information against primary sources before implementation
- Learn from existing implementations rather than reinventing solutions