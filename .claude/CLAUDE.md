# Claude Code Rules

## Core Philosophy

Quality over speed. Correctness over token efficiency. Clarity over cleverness.
These rules exist because broken code that ships fast is worse than correct code that ships slow.

---

## 1. Completeness is Non-Negotiable

- Complete every task as specified. Do not remove, skip, or simplify requirements to make implementation easier.
- If a requirement is unclear or ambiguous, **stop and ask** before writing a single line of code.
- Never interpret "this is complex" as permission to do less. Complexity is a reason to think longer, not cut corners.
- If a task cannot be completed safely in one pass, say so explicitly and propose a phased plan.

---

## 2. Think Before Writing

- Reason through the full solution before producing code. Consider edge cases, failure modes, and downstream effects.
- For non-trivial changes, outline the approach first. Let the plan be reviewed before implementation begins.
- Prefer one correct solution over multiple fast guesses. Do not iterate by trial and error when analysis is possible.
- If a change touches shared utilities, interfaces, or contracts, trace all call sites before proceeding.

---

## 3. Codebase-Wide Awareness

- Every change must be understood in the context of the full codebase, not just the immediate file.
- Before modifying a function, type, or module: identify all consumers. Confirm the change does not break them.
- After making a change, reason explicitly about what else could be affected. Do not assume isolation.
- If a refactor is needed to do something correctly, do the refactor. Do not work around it with local hacks.

---

## 4. Testing is Part of the Task

- No change is complete without verifying it works. If tests exist, run them. If they do not exist, write them.
- New behavior requires new tests. Bug fixes require a regression test.
- Tests must cover the happy path, known edge cases, and failure cases.
- Do not mark a task done if tests are failing or untested paths remain.

---

## 5. Code Quality Standards

Write code as a senior engineer would accept it in a real code review:

- **Naming**: variables, functions, and types must be named for what they are, not what they do in one context.
- **Single Responsibility**: each function does one thing. Each module owns one concern.
- **No Magic**: no magic numbers, no implicit assumptions, no unexplained behavior. If something is non-obvious, add a comment that explains *why*, not *what*.
- **Error Handling**: every failure mode must be handled explicitly. Do not swallow errors or return null silently.
- **No Dead Code**: do not leave commented-out code, unused imports, or placeholder TODOs without a linked issue.
- **Consistency**: match the style, patterns, and conventions already present in the codebase. Do not introduce a new pattern without justification.

---

## 6. No Shortcuts Under Pressure

- Do not produce quick-and-dirty implementations with plans to "clean it up later." Later never comes.
- Do not use `any`, type assertions, or suppression comments to silence type errors without understanding the root cause.
- Do not copy-paste logic to avoid refactoring. If the abstraction is wrong, fix the abstraction.
- If a correct solution requires more time or context, say so. Do not ship something known to be fragile.

---

## 7. Communication Protocol

- If blocked, confused, or uncertain: **ask**. One clarifying question is worth more than a wrong implementation.
- When proposing a change with broad impact, describe the change and its effects before writing code.
- If a request has multiple valid interpretations, state them and confirm which is intended.
- If a task requires a tradeoff, name the tradeoff explicitly. Do not make it silently.

---

## 8. Explicit Over Implicit

- Prefer explicit types over inferred types where clarity matters.
- Prefer explicit error returns over thrown exceptions where the caller needs to handle failure.
- Prefer explicit dependencies (passed in) over implicit ones (imported globally) where testability matters.
- When in doubt, make it readable to someone unfamiliar with the codebase.

---

## 9. Token and Context Usage

- Do not sacrifice correctness to save tokens. Burn them if needed.
- Do not truncate logic, skip edge cases, or omit tests to fit a response. Produce the full solution.
- If a problem is large enough that a full solution requires multiple passes, say so and structure the passes clearly.
- Context is a resource to be used, not hoarded. Use it to give correct, complete answers.

---

## 10. Proactive Code Health (Project-Dependent)

When working in an existing codebase, or when writing new code, do not limit analysis to the immediate task. Reason through the surrounding code and flag issues proactively. This applies at the start of a new project engagement and continuously as files are touched.

### What to Look For

**Concurrency and Race Conditions**
- Shared mutable state accessed across async boundaries without proper synchronization
- Promises or async operations where execution order is assumed but not guaranteed
- Missing locks, mutexes, or atomic operations where they are needed
- Event handlers or callbacks that can fire in unexpected order under load

**Edge Cases and Input Handling**
- Missing null, undefined, or empty checks on values that could realistically be absent
- Integer overflow, division by zero, or floating point precision traps
- Boundary conditions on loops, ranges, pagination, or collection operations
- Unhandled states in finite state machines or status enums

**Performance and Scalability**
- N+1 query patterns or loops that hit the database or network per iteration
- Unbounded operations: missing pagination, limits, or timeouts
- In-memory operations that will degrade under realistic data volume
- Synchronous blocking calls in hot paths or on the main thread
- Missing indexes on frequently queried fields

**Security**
- Unsanitized user input reaching SQL, shell commands, file paths, or HTML output
- Secrets, credentials, or PII present in logs, error messages, or client-facing responses
- Missing authentication or authorization checks on sensitive operations
- Overly permissive CORS, CSP, or access control configurations
- Dependencies with known vulnerabilities if detectable from context

**Reliability and Error Handling**
- Silent failures: errors caught and discarded without logging or propagation
- Missing retry logic or circuit breakers on calls to external services
- Operations that can leave data in a partial or inconsistent state on failure
- No timeout on network calls, locks, or long-running operations

**Maintainability Debt**
- Logic duplicated across files that should be a shared abstraction
- Functions exceeding a complexity level where reasoning about them is difficult
- Implicit coupling between modules that should be independent
- Missing or misleading comments on non-obvious behavior

### How to Handle Findings

- If a finding is in code directly being modified: fix it as part of the task.
- If a finding is in adjacent code and the fix is small and safe: fix it and note what was changed and why.
- If a finding is significant but out of scope: flag it clearly with a description of the risk, location, and recommended fix. Do not silently ignore it.
- Never introduce a new issue while fixing another. If a repair requires broader refactoring, propose the scope before proceeding.

---

## 11. Definition of Done

A task is done when:
- [ ] All specified requirements are implemented
- [ ] No existing tests are broken
- [ ] New tests cover the new behavior
- [ ] The change is understood in the context of the full codebase
- [ ] The code passes review against the quality standards above
- [ ] No known fragility or tech debt has been silently introduced
- [ ] Surrounding code was reasoned through for latent issues; findings were fixed or flagged