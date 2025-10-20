(rules:rule_4)=
# Rule 4: Start by Thinking Through a Potential Solution

Begin AI coding sessions by first working to understand and articulate the problem you're trying to solve, specified at a level of abstraction that makes it solvable by code, and think through how you anticipate it might be solved. Think through the entire problem space: What are the inputs and expected outputs? What are the key constraints and edge cases? What does success look like? This problem-focused approach serves a dual purpose: it helps you clarify exactly what you want the AI to accomplish so that you can evaluate its outputs appropriately, and it prevents the AI from making incorrect assumptions about your goals. When you provide problem context along with architectural details of how you anticipate a solution working (i.e., how it might fit in with the "bigger picture"), the AI generates code that fits naturally into your project rather than creating isolated solutions. Include details about data flow, component interactions, and expected interfaces when possible or if relevant. This approach transforms the AI from a code generator into an architecture-aware development partner. You can use LLMs to help generate externally-managed context files, and also look at GitHub Spec Kit for specification-driven workflows that define project requirements and gated phases (Specify, Plan, Tasks). AI can help you implement sophisticated patterns like structured checklists for iterative development that would be tedious to write from scratch.

## What separates positive from flawed examples

Flawed examples dump vague requirements on the AI without any planning. You haven't thought through how the pieces should fit together, what the data flow looks like, or what success actually means. The AI fills in the gaps with its own assumptions, which are probably wrong for your use case. You end up with code that technically does something but doesn't integrate well with your existing system.

Positive examples start with upfront planning. You've thought through the problem structure, anticipated how it should be solved, and specified the architectural context. You can articulate inputs, outputs, data flow, and how this component interacts with others. This gives the AI enough structure to generate code that actually fits your needs rather than creating a standalone solution that requires major rework.

---

### Example 1: Vague Requirements Without Planning

The user hasn't thought through what they actually need. No specification of how this integrates with existing code. No clarity on data flow or expected behavior. The AI has to guess at everything (data format, processing steps, output structure). What comes back is generic code that probably doesn't match the actual system architecture. When the user tries to integrate this, they'll discover all sorts of mismatches they should have specified upfront: the function expects a file path but the existing `DataLoader` returns a dictionary; the return type is ambiguous (NIfTI object? numpy array? dictionary?); there's no error handling for missing files or incompatible data; and the validation function is completely arbitrary because the user never specified what valid data looks like. The "modular" request gets interpreted as separate functions, but with no clear interfaces or type specifications, making them impossible to test or reuse independently.

````{prf:example} Dumping requirements without architectural context
:label: rule4-bad-ex1
:class: dropdown

**User Prompt:**
```
I need a preprocessing pipeline for fMRI data. It should do motion correction, 
normalization, and smoothing. Make it modular.
```

**AI Response:**
```python
def preprocess_fmri(input_file):
    """Preprocess fMRI data."""
    # Load data
    data = load_data(input_file)
    
    # Motion correction
    corrected = motion_correct(data)
    
    # Normalize
    normalized = normalize(corrected)
    
    # Smooth
    smoothed = smooth(normalized)
    
    return smoothed

def motion_correct(data):
    # Implementation
    pass

def normalize(data):
    # Implementation
    pass

def smooth(data):
    # Implementation  
    pass

def validate_data(data):
    # Some validation
    return True
```

**User tries to integrate:**
```python
# Existing code uses DataLoader that returns dict
loader = DataLoader()
images = loader.load('sub-01')  # Returns {'bold': NIfTI, 'T1w': NIfTI}

# But new function expects file path!
result = preprocess_fmri(images)  # TypeError: expected str, got dict

# Even if we fix that, what does it return?
# NIfTI object? numpy array? dict? Can't tell from code.
# And which image does it operate on? Both? Just BOLD?
```
````

---

### Example 2: Upfront Architecture Specification

The user has done the planning work before asking for code. They specify the complete system architecture with clear class boundaries and responsibilities. They define the exact API for each component including input/output types, error conditions, and side effects. They explain how data flows through the system and what each transformation does. They specify which parts are already implemented versus what needs to be created. This level of detail ensures the AI understands not just what to do, but how it should fit with existing code patterns and design. The resulting implementation can actually be dropped into the codebase because it matches the existing architecture.

````{prf:example} Clear architectural specification before implementation
:label: rule4-good-ex1
:class: dropdown

**User Prompt:**
```
I'm building a neuroimaging pipeline with these components:
- DataLoader class: reads BIDS-formatted data, returns NiBabel objects
- Preprocessor class: takes NiBabel objects, applies motion correction, 
  returns processed arrays
- Analyzer class: takes processed arrays, computes statistics

Here's the complete API structure:

class DataLoader:
    def load(self, subject_id: str) -> dict[str, nib.Nifti1Image]:
        '''
        Returns: {'T1w': Nifti1Image, 'bold': Nifti1Image}
        '''

class Preprocessor:
    def process(self, images: dict[str, nib.Nifti1Image]) -> np.ndarray:
        '''
        Main method to implement. Must call internal methods in sequence.
        Returns: 4D array (x, y, z, time) of preprocessed BOLD data
        '''
    
    def _motion_correct(self, img: nib.Nifti1Image) -> nib.Nifti1Image:
        '''
        Already implemented. Applies 6-parameter rigid body correction.
        Args: 
            img: 4D NIfTI image (x, y, z, time)
        Returns: 
            Motion-corrected NIfTI with same dimensions
        Raises:
            ValueError: if image is not 4D
        Side effects: 
            Sets self.motion_parameters (n_timepoints, 6) array
        '''
    
    def _normalize(self, img: nib.Nifti1Image, template: str = 'MNI152') -> nib.Nifti1Image:
        '''
        Already implemented. Warps to template space.
        Args:
            img: 3D or 4D NIfTI image
            template: One of ['MNI152', 'MNI305', 'Talairach']
        Returns:
            Normalized image in template space (may have different dimensions)
        Note: 
            Uses affine from self.normalization_matrix if available
        '''
    
    def _smooth(self, img: nib.Nifti1Image, fwhm: float = 6.0) -> np.ndarray:
        '''
        Already implemented. Applies Gaussian smoothing.
        Args:
            img: NIfTI image to smooth
            fwhm: Full-width half-maximum in mm
        Returns:
            Smoothed data as numpy array (loses NIfTI header info)
        Important: 
            Returns numpy array, not NIfTI object
        '''

class Analyzer:
    def compute_glm(self, data: np.ndarray, design: pd.DataFrame) -> StatsResult:
        '''
        Args:
            data: 4D array (x, y, z, time) from Preprocessor.process()
            design: Design matrix with columns as regressors, rows as timepoints
        Returns:
            StatsResult with attributes: .beta_maps, .t_maps, .p_values
        '''

Now implement Preprocessor.process() considering:
1. Must handle the 'bold' key from images dict
2. Chain the three internal methods correctly
3. Handle the type conversion from NIfTI to numpy array
4. Include proper error handling for missing keys
5. Preserve metadata needed for later analysis
```

**AI Response:**
```python
def process(self, images: dict[str, nib.Nifti1Image]) -> np.ndarray:
    """
    Process BOLD images through full preprocessing pipeline.
    
    Parameters
    ----------
    images : dict[str, nib.Nifti1Image]
        Dictionary with 'bold' key containing 4D fMRI data
        
    Returns
    -------
    np.ndarray
        Preprocessed 4D array (x, y, z, time)
        
    Raises
    ------
    ValueError
        If 'bold' key missing or image has wrong dimensions
    """
    if 'bold' not in images:
        raise ValueError("images dict must contain 'bold' key")
    
    bold_img = images['bold']
    
    # Validate 4D
    if bold_img.ndim != 4:
        raise ValueError(f"BOLD must be 4D, got {bold_img.ndim}D")
    
    # Step 1: Motion correction (returns NIfTI)
    corrected = self._motion_correct(bold_img)
    
    # Step 2: Normalization (returns NIfTI, may change dimensions)
    normalized = self._normalize(corrected, template='MNI152')
    
    # Step 3: Smoothing (returns numpy array)
    smoothed = self._smooth(normalized, fwhm=6.0)
    
    return smoothed
```
````

---

### Example 3: Iterative Refinement with Clear Goals

The user starts with a basic specification to establish the core functionality. Once that works, they add complexity through well-defined iterations. Each iteration has clear success criteria and builds naturally on the previous version. The user can verify each iteration independently before moving to the next. This prevents the AI from trying to solve everything at once and making wrong assumptions about priorities. The progressive refinement keeps the conversation focused and makes debugging much easier.

````{prf:example} Progressive specification with clear success criteria
:label: rule4-good-ex2
:class: dropdown

**User Prompt (Initial):**
```
Create a function to compute correlation between brain regions. Here's the initial API:

def compute_brain_correlation(
    data: np.ndarray,  # Shape: (n_regions, n_timepoints)
) -> np.ndarray:  # Shape: (n_regions, n_regions)
    '''Compute Pearson correlation between brain region timeseries'''

It should pass this basic test:
def test_basic_correlation():
    data = np.array([[1, 2, 3], [4, 5, 6]])
    corr = compute_brain_correlation(data)
    assert corr.shape == (2, 2)
    assert np.allclose(corr.diagonal(), 1.0)
```

**AI Response:**
```python
import numpy as np

def compute_brain_correlation(data):
    return np.corrcoef(data)
```

**User Follow-up:**
```
Good foundation. Now extend to handle missing data:

def compute_brain_correlation(
    data: np.ndarray,
    missing_method: str = 'pairwise'  # 'pairwise' or 'listwise'
) -> np.ndarray:

Add this test:
def test_missing_data():
    data = np.array([[1, 2, np.nan, 4], [2, 4, 6, 8]])
    corr_pair = compute_brain_correlation(data, 'pairwise')
    corr_list = compute_brain_correlation(data, 'listwise')
    # Pairwise uses available pairs, listwise drops entire timepoint
    assert not np.isnan(corr_pair[0, 1])
    assert np.isnan(corr_list[0, 1])
```

**AI Response:**
```python
import numpy as np
import pandas as pd

def compute_brain_correlation(data, missing_method='pairwise'):
    if missing_method == 'pairwise':
        df = pd.DataFrame(data.T)
        return df.corr().values
    elif missing_method == 'listwise':
        mask = ~np.isnan(data).any(axis=0)
        clean_data = data[:, mask]
        if clean_data.shape[1] < 2:
            return np.full((data.shape[0], data.shape[0]), np.nan)
        return np.corrcoef(clean_data)
    else:
        raise ValueError(f"Unknown method: {missing_method}")
```

**User:**
```
Perfect. Final iteration - add Fisher z-transformation:

def compute_brain_correlation(
    data: np.ndarray,
    missing_method: str = 'pairwise',
    fisher_z: bool = False
) -> dict:  # Returns dict with 'correlation', 'p_values', 'z_scores' keys

Test:
def test_fisher_transform():
    data = np.random.randn(10, 100)
    result = compute_brain_correlation(data, fisher_z=True)
    r = result['correlation'][0, 1]
    z_expected = 0.5 * np.log((1 + r) / (1 - r))
    assert np.allclose(result['z_scores'][0, 1], z_expected)
```
````
