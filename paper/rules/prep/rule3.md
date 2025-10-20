(rules:rule_3)=
# Rule 3: Choose Appropriate AI Interaction Models

It's tempting to use the AI tools to independently generate a complete codebase, but one quickly ends up being divorced from the code. A pair programming model, where one directs IDE-based AI assistants through comments in the code, is a way to stay in close touch with the code. Different interaction paradigms (including IDE-integrated assistants, conversational interfaces, and autonomous coding agents) offer distinct advantages for different types of development tasks. Match AI tool capabilities with task requirements, developer preferences, and project constraints. A summary of different interaction paradigms, as well as their strengths and limitations in 2025, are provided in Table 1.

**Table 1: Comparison of AI coding tool interaction models**

AI-assisted development tools are categorized by interaction model and deployment scenario. Each paradigm offers distinct advantages for different phases of software development, with trade-offs between automation level and developer control.

| Tool Type | Best For | Description |
|-----------|----------|-------------|
| Conversational (ChatGPT, ### Example, etc.) | Architecture design, complex debugging, learning new concepts | Deep reasoning and flexible problem-solving with extensive context handling, but requires manual code transfer and loses context between sessions |
| IDE Assistant (CoPilot, IntelliSense, etc.) | Code completion, refactoring, maintaining flow | Seamless workflow integration with immediate feedback and preserved code context, but limited reasoning for complex architectural decisions |
| Autonomous Agents (### Example Code, Cursor, Aider, etc.) | Rapid prototyping, multi-file changes, large refactoring | High-speed implementation that can work independently across multiple files, but risks code divergence and requires careful monitoring |


## What separates positive from flawed examples

Flawed examples use the wrong tool for the job. You might use a conversational interface for tasks requiring tight integration with your codebase, or an autonomous agent for exploratory work where you need to understand every step. The mismatch leads to wasted time, loss of context, or code you don't understand.

Positive examples deliberately choose interaction models based on the task. Quick exploratory analysis? Conversational interface works fine. Complex refactoring across multiple files? IDE assistant keeps you in the code. Building a new feature with clear specs? Autonomous agent can handle it while you verify the approach. The key is maintaining appropriate oversight for the task complexity.

---

## Example 1: Wrong Tool for the Job

Using a conversational interface for complex codebase modifications creates multiple integration problems. You lose context of the full codebase structure since the AI only sees what you paste. The AI regenerates entire code blocks that you have to manually copy back into files, with no clear indication of what changed versus what stayed the same. This breaks your git history and makes it impossible to review changes properly. You can't easily test the modification in isolation, and you have to manually figure out import statements and dependencies. Worse, you lose track of how the change integrates with other parts of your system (in this case, how skull-stripping affects QualityControl reports). By the time you're done, you have to manually verify the entire class still works rather than just verifying the new feature.

````{prf:example} Using conversational interface for complex codebase integration
:label: rule3-bad-ex1
:class: dropdown

**User Prompt (in ChatGPT):**
```
I need to add a new preprocessing step to my neuroimaging pipeline. The pipeline has:
- DataLoader class that reads BIDS data
- Preprocessor class with motion correction, normalization, smoothing
- QualityControl class that generates reports
- Main script that orchestrates everything

Add skull-stripping before normalization. Here's my current Preprocessor code:
[pastes 200 lines of code]
```

**AI Response:**
```python
# Updated Preprocessor class
class Preprocessor:
    def __init__(self, ...):
        # [AI regenerates entire class with skull-stripping added]
    
    def process(self, ...):
        # [150 lines of modified code]
    
    def _skull_strip(self, img):
        # [new method]
        # ...
```
````

---

## Example 2: IDE Assistant for Codebase Integration

The IDE maintains full context of existing code automatically. Changes are made in place in actual files. You can immediately see what's different with a git diff. Integration with existing methods is preserved naturally. Import statements get suggested automatically as you need them. You can run tests immediately to verify everything still works. This maintains your natural development workflow. Most importantly, you stay oriented in the codebase throughout the entire process instead of copying and pasting code between a chat window and your editor.

````{prf:example} Using IDE-integrated assistant for in-context modification
:label: rule3-good-ex1
:class: dropdown

**Context:** Same task (add skull-stripping), but using IDE assistant (GitHub Copilot, Cursor, etc.)

**User Approach:**
```python
# In Preprocessor.py file, user adds comment before _normalize method:

class Preprocessor:
    def process(self, images):
        """Process images through pipeline."""
        bold_img = images['bold']
        
        # Apply motion correction
        corrected = self._motion_correct(bold_img)
        
        # ADD: Skull-strip the motion-corrected image before normalization
        # Use FSL's BET algorithm with default parameters
        # Input: 3D or 4D NIfTI image
        # Output: skull-stripped NIfTI image
        # Should preserve image dimensions and affine
        
        # [Copilot suggests implementation here]
```

**IDE Assistant Response:**
```python
        # Skull-strip before normalization
        stripped = self._skull_strip(corrected)
        
        # Apply normalization
        normalized = self._normalize(stripped, template='MNI152')
```

**User scrolls down, adds method signature:**
```python
    def _skull_strip(self, img):
        """
        Remove skull from brain image using FSL BET.
        
        Parameters
        ----------
        img : nibabel.Nifti1Image
            Motion-corrected image (3D or 4D)
            
        Returns
        -------
        stripped : nibabel.Nifti1Image
            Skull-stripped image with same dimensions
        """
        # [Copilot generates implementation using nipype BET interface]
```
````

---

## Example 3: Autonomous Agent for Well-Specified Feature

The task is well-defined and isolated, making it suitable for autonomous development. Clear specifications and acceptance criteria guide the agent. The agent has enough context (existing code style) to match conventions. The user maintains oversight by reviewing everything before integration. Task complexity matches what agents can actually handle reliably. Final verification ensures quality before accepting anything into the codebase. The key is that this feature can be developed somewhat independently, reviewed as a unit, and integrated deliberately.

````{prf:example} Using autonomous agent for isolated feature development
:label: rule3-good-ex2
:class: dropdown

**User Prompt (to ### Example Code or similar agent):**
```
Create a new module `pipeline/quality_control.py` that generates QC reports for neuroimaging preprocessing. 

**Requirements:**
- Function generate_qc_report(preprocessed_data, output_dir)
- Input: dict with keys 'original', 'motion_corrected', 'normalized', 'smoothed'
- Each value is a nibabel.Nifti1Image object

**Report should include:**
1. Before/after montages (original vs. final)
2. Motion parameter plots (translation, rotation over time)
3. SNR calculations (mean/std in brain mask)
4. Framewise displacement timeseries
5. Output as HTML report with embedded images

**Code structure:**
- Use plotly for interactive plots
- Use nilearn for brain visualizations
- Follow existing code style in pipeline/preprocessor.py
- Include comprehensive docstrings
- Create tests in tests/test_quality_control.py

**Dependencies:**
Can use: nibabel, nilearn, plotly, numpy, pandas
Must be compatible with existing pipeline in pipeline/preprocessor.py

I'll review the implementation before integrating.
```

**Agent Actions:**
```
1. Reads pipeline/preprocessor.py to understand existing style
2. Creates pipeline/quality_control.py with complete implementation
3. Creates tests/test_quality_control.py with unit tests
4. Generates example HTML report in examples/
5. Updates requirements.txt with new dependencies
6. Creates documentation in docs/quality_control.md
```

**User Review:**
```
- Checks implementation matches specification
- Reviews test coverage
- Runs tests to verify functionality
- Examines example report output
- Verifies integration points with existing pipeline
- Provides feedback on any issues before accepting
```
````

---

## Example 4: Conversational Interface for Exploratory Analysis

This is a quick exploratory task, not building production code. The conversational back-and-forth helps understand data issues naturally. You don't need to integrate this exploration into your codebase at all. You can iterate quickly on visualization and analysis without worrying about code style or test coverage. Once you understand what you're dealing with, you can write the proper pipeline with whichever tool makes sense for that. The exploration stays in the chat where it belongs, and the real implementation goes into your codebase properly.

````{prf:example} Using chat interface for quick data exploration
:label: rule3-good-ex3
:class: dropdown

**Context:** Exploring a new dataset before implementing pipeline

**User Prompt:**
```
I have a CSV file with brain connectivity data (attached: connectivity_matrix.csv). 
Before I build my analysis pipeline, help me understand the data structure:

1. What are the dimensions?
2. Are there any NaN or infinite values?
3. What's the distribution of values?
4. Are there any suspicious patterns (e.g., all zeros in certain rows)?
5. Suggest appropriate preprocessing steps
```

**AI Response:**
```python
import pandas as pd
import numpy as np

# Load and examine
data = pd.read_csv('connectivity_matrix.csv', index_col=0)

print(f"Shape: {data.shape}")
print(f"NaN count: {data.isna().sum().sum()}")
print(f"Infinite values: {np.isinf(data.values).sum()}")
print(f"\nValue range: [{data.min().min():.3f}, {data.max().max():.3f}]")
print(f"Mean: {data.mean().mean():.3f}")

# Check for problematic patterns
zero_rows = (data == 0).all(axis=1).sum()
print(f"\nAll-zero rows: {zero_rows}")

# Visualize distribution
import matplotlib.pyplot as plt
plt.hist(data.values.flatten(), bins=50)
plt.title("Distribution of connectivity values")
plt.show()
```

**Findings:**
```
Shape: (100, 100)
NaN count: 245
Infinite values: 0

Value range: [-0.856, 0.923]
Mean: 0.234

All-zero rows: 3

Distribution appears bimodal - suggests two different connectivity regimes
```

**User:** "Why would I have all-zero rows?"

**AI:** "Likely those brain regions were excluded during preprocessing (e.g., due to artifacts)..."
````
