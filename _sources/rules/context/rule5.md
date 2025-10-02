(rules:rule_5)=
# Rule 5: Manage Context Strategically

Context is everything in AI-assisted coding. Provide all necessary information upfront through clear documentation, attached references, or structured project files with dependencies included. Don't assume the AI retains perfect context across long conversations; explicitly restate critical requirements, constraints, and dependencies when interactions get complex. Keep track of context and clear or compact when it's getting close to limits. Use memory files to keep important context available across sessions while minimizing irrelevant details that can degrade AI performance. Agents can effectively use memory files to keep important things in context for every session. It's also useful to keep a problem solving file, where you can add problems whenever you notice them, and where the model can keep track of its progress.

## What separates positive from flawed examples

Flawed examples assume the AI remembers everything from earlier in the conversation or from previous sessions. You add requirements incrementally without restating the full context. You hit context limits without noticing and the AI starts forgetting critical details. The conversation becomes polluted with failed attempts that confuse future interactions. You don't have any persistent way to carry important project information across sessions.

Positive examples treat context as a limited resource that needs active management. You provide complete context upfront when starting new tasks. You explicitly restate requirements when conversations get long. You use memory files to persist critical information across sessions. You recognize when context is polluted with failed attempts and restart cleanly. You track what the AI knows versus what it needs to be reminded about.

---

## Example 1: Incremental Requirements Without Context Management

The user keeps adding requirements one at a time without ever providing the full picture. The AI is trying to work with fragments of information spread across multiple messages. By the fifth requirement, the AI has probably lost track of earlier constraints. There's no way to know if all the requirements are actually compatible with each other. When this inevitably breaks, the user will have to either restart from scratch or spend time figuring out which requirements got lost. No memory file means next session starts from zero.

````{prf:example} Drip-feeding requirements across long conversation
:label: rule5-bad-ex1
:class: dropdown

**Message 1:**
```
Create a function to load neuroimaging data.
```

**Message 5 (after some back and forth):**
```
Oh, it also needs to handle BIDS format.
```

**Message 10:**
```
Actually, we need to support both NIfTI and GIFTI files.
```

**Message 15:**
```
Can you add validation for the BIDS structure?
```

**Message 20:**
```
Also, it needs to work with both local files and S3 buckets.
```

**Message 25:**
```
Wait, I forgot to mention it needs to handle compressed files too.
```

**Message 30:**
```
And it should cache the validation results.
```

[AI is now working with fragmented context spread across 30 messages, 
probably lost track of earlier requirements, no clear specification of 
what the complete system should be]
````

---

## Example 2: Complete Context Upfront with Memory File

The user provides everything the AI needs to know in the initial message. Complete requirements, constraints, existing code structure, file formats, all specified upfront. The AI can see the full problem and generate a coherent solution. The memory file captures architectural decisions and constraints that should persist across sessions. When the user starts a new session next week, they can reference the memory file and the AI immediately knows the project context. This prevents having to re-explain the entire system every time.

````{prf:example} Comprehensive initial specification with persistent context
:label: rule5-good-ex1
:class: dropdown

**User Prompt:**
```
I need to implement a neuroimaging data loader. Here's the complete context:

PROJECT ARCHITECTURE (see attached memory_file.md):
- Module: src/data/loaders.py
- Integrates with existing Pipeline class
- Returns standardized DataContainer objects

COMPLETE REQUIREMENTS:
1. Support formats: NIfTI (.nii, .nii.gz), GIFTI (.gii)
2. Handle BIDS-formatted datasets
3. Support both local filesystem and S3 (boto3)
4. Validate BIDS structure before loading
5. Handle compressed files transparently
6. Cache validation results (use joblib)
7. Thread-safe for parallel loading

DATA SPECIFICATIONS:
- BIDS dataset structure per v1.9.0 spec
- Required metadata: subject_id, session, task, run
- Optional metadata: space, desc, suffix
- Must preserve NIfTI affine transformations
- GIFTI should load all data arrays

ERROR HANDLING:
- Raise BIDSValidationError for invalid structure
- Raise FileNotFoundError with helpful message for missing files
- Raise ValueError for unsupported file formats
- Log warnings for missing optional metadata

INTEGRATION:
class DataContainer:
    '''Already exists in src/data/containers.py'''
    def __init__(self, data: np.ndarray, metadata: dict, affine: np.ndarray = None)

DEPENDENCIES:
- nibabel (for NIfTI/GIFTI)
- boto3 (for S3)
- joblib (for caching)
- bids-validator (for validation)

CONSTRAINTS:
- No external network calls during validation (offline mode)
- Maximum file size: 10GB
- Must work with Python 3.9+

Create class BIDSLoader with methods:
- load(subject_id, session=None) -> DataContainer
- validate_structure() -> bool
- list_subjects() -> list[str]
```

**Attached memory_file.md:**
```markdown
# Neuroimaging Pipeline - Data Loading

## Architecture Decisions
- All loaders return DataContainer objects for standardization
- S3 access uses boto3 with credentials from environment
- Caching uses joblib with 7-day TTL
- Thread-safety required for parallel subject processing

## Constraints
- No pandas dependencies (too heavy for this module)
- Must support offline validation
- File size limit: 10GB per file

## Known Issues
- NIfTI headers can be corrupted, always validate affine
- BIDS validator is strict, may reject valid legacy datasets
```
````

---

## Example 3: Context Reset When Conversation Gets Polluted

The conversation has gone off track with multiple failed attempts. Rather than continuing to patch a fundamentally broken approach, the user recognizes the context is polluted. They review what went wrong, update the memory file with lessons learned, and start fresh. The new conversation begins with corrected requirements and avoids the previous mistakes. This is much more efficient than trying to untangle a confused conversation with 30+ messages of failed attempts.

````{prf:example} Strategic restart with updated memory file
:label: rule5-good-ex2
:class: dropdown

**After 25 messages of failed attempts:**

**User Analysis:**
```
This conversation is polluted. Let me review what went wrong:

PROBLEMS IDENTIFIED:
1. Never specified data format upfront (AI assumed wrong structure)
2. Added requirements incrementally (created conflicts)
3. Didn't specify integration with existing Pipeline class
4. Edge cases discovered too late (should have been in requirements)

LESSONS LEARNED (adding to memory_file.md):
- fMRI data is 4D (x, y, z, time), not 3D
- Must preserve temporal ordering
- Existing Pipeline expects dict[str, np.ndarray] not list
- Need to handle motion-corrected vs raw data differently
```

**Updated memory_file.md:**
```markdown
# Neuroimaging Pipeline - Preprocessing Module

## Data Formats (CRITICAL)
- fMRI: 4D arrays (x, y, z, time) - NEVER assume 3D
- Temporal dimension must be preserved in exact order
- Motion correction creates alignment parameters (n_timepoints, 6)

## Integration Requirements
- Pipeline.process() expects: dict[str, np.ndarray]
- Keys: 'bold' (4D data), 'motion_params' (2D array), 'mask' (3D array)
- NOT a list or single array

## Edge Cases
- Some timepoints may be censored (marked with NaN)
- Motion params might be missing (use identity if so)
- Masks can have different resolutions than data
```

**New Session (Next Day):**
```
Starting fresh. Context from memory_file.md:
- fMRI is 4D (x, y, z, time)
- Pipeline expects dict with 'bold', 'motion_params', 'mask' keys
- Handle NaN for censored timepoints

Implement Preprocessor.load_and_prepare() that:
[Clear, complete specification based on lessons learned]
```
````

---

## Example 4: Using Problem Tracking File for Complex Debugging

The user maintains a problem tracking file that both they and the AI can reference. As issues are discovered, they get logged with context. The AI can see all known issues and their relationships. This prevents repeatedly fixing the same bug in different places or introducing fixes that conflict with each other. The problem file serves as a shared workspace that persists across sessions and keeps everyone aligned on what needs attention.

````{prf:example} Persistent problem tracking across sessions
:label: rule5-good-ex3
:class: dropdown

**problems.md (maintained across sessions):**
```markdown
# Known Issues - Preprocessing Pipeline

## ACTIVE
1. **NaN handling inconsistent** (Priority: HIGH)
   - File: src/preprocessing/motion.py:45
   - Problem: _motion_correct() propagates NaN, but _normalize() fails on NaN
   - Impact: Pipeline crashes on censored timepoints
   - Attempted: Filling with zeros (broke downstream stats)
   - Next: Try masked arrays for NaN propagation
   
2. **Memory leak in batch processing** (Priority: MEDIUM)
   - File: src/preprocessing/batch.py:120
   - Problem: Preprocessor instances not garbage collected
   - Impact: Memory grows linearly with batch size
   - Reproduction: Process >50 subjects sequentially
   - Investigation: Need to profile __del__ method

3. **Affine transformations lost** (Priority: HIGH)
   - File: src/preprocessing/normalize.py:78
   - Problem: _normalize() returns array instead of NIfTI
   - Impact: Can't map results back to subject space
   - Root cause: Using np.array() instead of preserving NIfTI object
   
## RESOLVED
- ✓ Header corruption on S3 load (Fixed: added integrity check)
- ✓ Thread-safety issue in cache (Fixed: added lock)
```

**User Prompt (new session):**
```
Looking at problems.md, let's tackle issue #1 (NaN handling). 

Current situation:
- motion_correct() keeps NaN (correct for censored timepoints)
- normalize() crashes on NaN input
- Tried filling zeros, broke statistics

Proposed solution:
- Use numpy masked arrays throughout pipeline
- Preserve mask through all transformations
- Convert back to NaN at final output

Implement changes to normalize() and smooth() methods to handle masked arrays.
Keep the NaN preservation behavior in motion_correct().

After implementation, update problems.md to mark #1 as resolved and document 
the solution.
```
````
