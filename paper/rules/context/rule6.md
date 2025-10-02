(rules:rule_6)=
# Rule 6: Monitor Progress and Know When to Restart

It's tempting to just walk away and let the model work for a long time, but often the model will end up going down the wrong path, wasting time and tokens. You need to actively monitor what the AI is doing: Is it changing things you didn't want changed? Is it ignoring the changes you actually requested? Is it introducing new problems while trying to fix old ones? When you notice the AI heading in the wrong direction, stop it immediately rather than letting it continue down an unproductive path.

Sometimes the most efficient approach is recognizing when a conversation has become too convoluted with failed attempts and conflicting corrections. When this happens, review your prompt history to identify what went wrong: Were requirements unclear? Did you add conflicting constraints? Did you forget to specify critical details upfront? Starting fresh with these lessons learned often produces better results than continuing to debug within a polluted context. Clear context and restart from memory files after adding additional details to prevent the same problem from occurring in the future. This also highlights the need for good version control; if you commit the code before undertaking a major change, it's easy to simply revert to the previous commit and start over if the model goes astray. Fortunately coding agents are generally very good at writing detailed commit messages, making a commit as easy as typing "commit to git".

## What separates positive from flawed examples

Flawed examples let the AI run unsupervised, assuming it will eventually figure things out. You come back to find it's been making changes you didn't want, introducing new bugs while fixing old ones, or heading down a completely wrong architectural path. By the time you notice, you've wasted significant time and tokens on code you can't use. The conversation is now polluted with failed approaches that will confuse future attempts.

Positive examples involve active monitoring and decisive intervention. You watch what the AI is doing and stop it when it goes off track. You recognize when a conversation has gotten too tangled and needs a fresh start. You review what went wrong, update your specifications or memory files, and begin again with better context. You use version control to make restarts cheap and safe.

---

## Example 1: Unsupervised AI Goes Off Track

The user gives vague instructions and walks away. The AI makes assumptions about what changes are needed. By the time the user checks back, the AI has modified core functionality that shouldn't have been touched, introduced new dependencies without asking, changed the API in breaking ways, and created bugs in previously working code. The user now has to figure out what changed, what broke, and how to salvage any useful parts. Much of the work is unusable and the conversation is polluted with bad approaches.

````{prf:example} Letting AI work unsupervised with vague instructions
:label: rule6-bad-ex1
:class: dropdown

**User Prompt:**
```
Fix the preprocessing pipeline. It's got some bugs and could be more efficient.
I'll be back in an hour.
```

[User leaves for an hour]

**AI Makes Changes:**
```python
# Changed _motion_correct to use different algorithm
def _motion_correct(self, img):
    # Switched from SPM to FSL without asking
    # Changed parameter defaults
    # Modified return format
    
# "Optimized" _normalize by removing validation
def _normalize(self, img):
    # Removed input validation "for efficiency"
    # Changed coordinate system handling
    # Now returns different shape than before
    
# "Improved" error handling
def process(self, images):
    # Wrapped everything in try/except that swallows errors
    # Added logging that breaks in production
    # Changed the API signature
```

**User Returns:**
```
Wait, what did you change?
- Why is motion correction using FSL now? We use SPM.
- Why did you remove input validation?
- The API signature changed and broke downstream code.
- New errors are being silently swallowed.
- You added a dependency on a logging framework we don't use.

I can't use any of this. Now I have to figure out what was actually wrong.
```
````

---

## Example 2: Active Monitoring with Early Intervention

The user stays engaged and watches what the AI is doing in real time. When the AI starts modifying the wrong function, the user stops it immediately. They clarify what should and shouldn't be changed. When the AI proposes a solution that doesn't match the architecture, the user catches it before implementation. This prevents wasted work and keeps the conversation focused on productive changes.

````{prf:example} Real-time monitoring with course correction
:label: rule6-good-ex1
:class: dropdown

**User Prompt:**
```
The preprocessing pipeline has a bug in _smooth() where it doesn't handle 
edge cases properly. Fix ONLY the _smooth() method. Do not modify 
_motion_correct() or _normalize().

Show me your proposed changes before implementing.
```

**AI Response:**
```python
# Proposed changes to _smooth():
def _smooth(self, img, fwhm=6.0):
    # Add validation for edge cases
    if img is None:
        raise ValueError("Image cannot be None")
    
    if fwhm <= 0:
        raise ValueError("FWHM must be positive")
    
    # Handle 3D vs 4D images
    if img.ndim == 3:
        return gaussian_filter(img, sigma=fwhm/2.355)
    elif img.ndim == 4:
        # Smooth each volume separately
        return np.stack([gaussian_filter(vol, sigma=fwhm/2.355) 
                        for vol in img])
    else:
        raise ValueError(f"Expected 3D or 4D image, got {img.ndim}D")
```

**User:**
```
Good, but you're using gaussian_filter from scipy. We use nilearn's 
smooth_img to preserve NIfTI metadata. Revise to use:

from nilearn.image import smooth_img
smoothed = smooth_img(img, fwhm=fwhm)

Also, the 3D case shouldn't happen in our pipeline - all BOLD data is 4D. 
Remove that branch and add a check that raises an error for 3D input.
```

**AI Response:**
```python
from nilearn.image import smooth_img

def _smooth(self, img, fwhm=6.0):
    if img is None:
        raise ValueError("Image cannot be None")
    
    if fwhm <= 0:
        raise ValueError("FWHM must be positive")
    
    # Verify 4D input
    if img.ndim != 4:
        raise ValueError(f"Expected 4D BOLD image, got {img.ndim}D. "
                        f"All BOLD data should be 4D (x, y, z, time).")
    
    return smooth_img(img, fwhm=fwhm)
```

**User:**
```
Perfect. Implement that change to _smooth() only. Do not modify any other methods.
```
````

---

## Example 3: Recognizing When to Restart

After several attempts to fix a complex bug, the conversation has become tangled with conflicting constraints and failed approaches. Rather than continuing to patch, the user recognizes this is a restart situation. They commit the current (working) code, review what went wrong, update memory files with lessons learned, and start a fresh conversation with better specifications. The new attempt succeeds quickly because the requirements are now clear and the context isn't polluted.

````{prf:example} Strategic restart after conversation becomes polluted
:label: rule6-good-ex2
:class: dropdown

**After 20+ messages of failed attempts:**

**User:**
```
This conversation is too polluted. Let me step back and figure out what went wrong.

REVIEWING PROMPT HISTORY:
1. Started vague: "Fix the normalization issues"
2. Added constraint: "Actually, preserve the original space"
3. Conflicting: "Wait, also support template space"
4. Late addition: "Oh, it needs to work with pediatric data too"
5. Contradiction: "Never mind, adult templates only"

PROBLEMS IDENTIFIED:
- Requirements were unclear from the start
- Added conflicting constraints incrementally
- Never specified the actual use case
- AI tried to accommodate all contradictions, creating mess

LESSONS LEARNED:
- Normalization needs TWO separate code paths: original space vs template
- Pediatric vs adult is a data issue, not a code issue
- Need to specify WHICH space for WHICH analyses upfront

Committing current working code, then starting fresh.
```

**Git commit:**
```bash
git add .
git commit -m "Working preprocessing before normalization refactor"
```

**Updated memory_file.md:**
```markdown
# Preprocessing Pipeline - Normalization

## Requirements (CLARIFIED)
- Two separate normalization modes:
  1. intensity_only: normalize intensity, preserve native space
  2. spatial: normalize to MNI152 template space
- Mode determined by downstream analysis type
- NO attempt to support both simultaneously

## Use Cases
- intensity_only: for ROI-based analyses in native space
- spatial: for group-level voxelwise analyses

## Data Constraints
- Adult data only (18-65 years)
- Pediatric data requires different pipeline entirely
```

**New Session (Fresh Start):**
```
Context from memory_file.md: Need two normalization modes (intensity vs spatial).

Implement _normalize() with clear separation:

def _normalize(self, img: nib.Nifti1Image, mode: str = 'intensity') -> nib.Nifti1Image:
    '''
    Args:
        mode: 'intensity' (preserve space) or 'spatial' (warp to MNI152)
    '''
    
Test cases:
1. intensity mode: output space matches input space
2. spatial mode: output is in MNI152 space
3. invalid mode: raises ValueError

Do NOT try to support both modes simultaneously or add pediatric handling.
```
````

---

## Example 4: Using Version Control for Safe Restarts

The user commits working code before attempting major changes. When the AI's approach turns out to be wrong, reverting is trivial. This makes restarts essentially free. The detailed commit messages (which coding agents excel at writing) make it easy to track what was attempted and why. Version control transforms restarts from costly failures into cheap experiments.

````{prf:example} Version control enables low-cost restarts
:label: rule6-good-ex3
:class: dropdown

**User:**
```
Current code works but is slow. Going to try optimization. 
Committing first so we can easily revert if needed.
```

**Git:**
```bash
git add preprocessing/pipeline.py
git commit -m "Working preprocessing pipeline - baseline before optimization

Current performance:
- Processing time: ~45 seconds per subject
- Memory usage: ~2GB peak
- Works correctly on all test cases

Planning to optimize motion correction step which takes 30s of the 45s total."
```

**User to AI:**
```
Optimize _motion_correct() for speed. Current baseline: 30 seconds per subject.
Target: under 10 seconds. Maintain correctness on existing test suite.
```

**AI attempts optimization, but:**
```python
# New version is faster but fails edge cases
# Introduced numerical instability
# Doesn't handle censored timepoints correctly
```

**User:**
```
This optimization broke correctness. Reverting.
```

**Git:**
```bash
git diff  # Review what changed
git checkout preprocessing/pipeline.py  # Revert to working version
```

**User:**
```
Let's try a different approach. The problem was trying to vectorize 
the entire algorithm. Instead, let's just parallelize across timepoints.

[Fresh conversation with better optimization strategy]
```
````

---

**Key Takeaways:**
- Actively monitor AI's work rather than walking away
- Stop the AI immediately when it goes off track
- Recognize when conversations become too polluted to continue
- Review what went wrong before restarting
- Update memory files with lessons learned
- Use version control to make restarts cheap and safe
- Commit before major changes so you can easily revert
- A fresh start with better context beats debugging in polluted conversations