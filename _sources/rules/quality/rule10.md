(rules:rule_10)=
# Rule 10: Refine Code Incrementally with Focused Objectives

Once you have working, tested code, resist the temptation to ask AI to "improve my codebase." Instead, approach refinement incrementally with clear, focused objectives. Be explicit about what aspect you want to improve: performance optimization, code readability, error handling, modularity, or adherence to specific design patterns. When you recognize that refinement is needed but can't articulate the specific approach (for instance, you know certain logic should be extracted into a separate function but aren't sure how), use AI to help you formulate concrete objectives before implementing changes. Describe what you're trying to achieve and ask the AI to suggest specific refactoring strategies or design patterns that would accomplish your goal, applying the same mindsets delineated in Rules 1 - 9 to help you along the way.

AI excels at identifying opportunities for refactoring and abstraction, such as recognizing repeated code that should be extracted into reusable functions or methods, and detecting poor programming patterns like deeply nested conditionals, overly long functions, tight coupling between components, sloppy or inconsistent variable naming conventions, and other poor patterns. When requesting refinements, specify the goal (e.g., "extract the data validation logic into a separate function" rather than "make this better") and verify each change against your tests (or, improving your testing as you interate to reflect the latest updates and improvements) before moving to the next improvement. This focused approach prevents the AI from making changes that, while technically sound, don't align with your project's architectural decisions. Note that AI can inadvertently break previously working code or degrade performance while making stylistic improvements. Always test thoroughly after each incremental change, and revert if the "improvement" introduces problems or doesn't provide clear benefits. 

## What separates positive from flawed examples

Flawed examples ask AI to "improve" or "clean up" code without specific objectives. The AI makes sweeping changes across multiple concerns simultaneously—renaming variables, restructuring logic, changing algorithms, adding abstractions. You can't evaluate which changes are beneficial because everything changed at once. Tests start failing but you don't know which modification caused the problem. The AI might introduce technically correct patterns that don't match your project's conventions. You waste time untangling good changes from bad ones, or worse, accept problematic changes because you can't isolate their effects.

Positive examples approach refinement systematically. You either identify specific issues yourself or ask AI to diagnose problems first, then evaluate its suggestions based on your project context. You tackle one focused objective at a time. You test after each change (improving your testing suite as you go) and revert immediately if something breaks. You recognize that not all AI suggestions are appropriate for your codebase; even good practices can be wrong if they conflict with project or field conventions for specific tendencies or methodologies. This incremental approach lets you understand each change, verify its benefit, and maintain a working codebase throughout refinement.

---

## Example 1: Vague "Improve This" Request

The user asks AI to generically improve code without specific objectives. The AI makes sweeping changes across multiple dimensions. When tests fail, the user tries to salvage the situation by listing what's wrong, but the AI's attempts to fix the problems make things worse. The conversation becomes increasingly polluted with failed attempts, conflicting constraints, and mounting confusion. What started as "improve the code" turns into a debugging nightmare where it's impossible to tell what's broken, why, or how to fix it. The user eventually gives up and has to revert everything.

````
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

