# Development Workflow

This repository uses an issue-driven, PR-first workflow. Direct pushes to
`main` are reserved for repository administration only. Feature work, fixes,
documentation updates, and compiler design changes should all go through a
branch and pull request.

## Goals

- Keep every change traceable to a GitHub issue.
- Review compiler behavior, tests, and docs before merging.
- Avoid mixing unrelated requirements in one branch.
- Preserve `main` as a working baseline for local users and CI.

## Required Flow

1. Create or select a GitHub issue.
2. Confirm the issue has a clear scope, acceptance criteria, and test plan.
3. Create a branch from the latest `main`.
4. Make the smallest coherent change that satisfies the issue.
5. Run the required local checks.
6. Open a pull request and link the issue.
7. Merge only after review approval and passing checks.

## Issue Requirements

Every implementation issue should include:

- **Problem:** what is currently broken, missing, or unclear.
- **Scope:** which subsystem is expected to change.
- **Out of scope:** related work that should not be included.
- **Acceptance criteria:** observable behavior required before closing.
- **Validation:** commands, tests, examples, or docs used to prove completion.

If the issue changes a compiler contract, IR format, runtime API, operator
extension path, pass pipeline, target/device semantics, or profiling output, it
must also include a design note before implementation starts.

## Branch Naming

Use short, issue-linked branch names:

```text
issue-<number>-<short-topic>
```

Examples:

```text
issue-1-cpu-build-smoke
issue-2-op-extension-contract
issue-14-hot-cold-adaptive-compile
```

## Commit Scope

Commits should be reviewable and focused:

- Keep unrelated issues on separate branches.
- Do not reformat files unrelated to the issue.
- Do not rewrite generated artifacts unless the issue requires it.
- Do not revert another user's local or remote changes without explicit
  agreement.
- Prefer one meaningful commit per completed issue unless incremental commits
  make review clearer.

## Pull Request Requirements

Every PR must include:

- Linked issue, using `Closes #<number>` when the PR completes the issue.
- Summary of behavior changed.
- Test commands and results.
- Risk notes for compiler/runtime behavior.
- Documentation updates, or a statement that no docs are needed.

For compiler-facing changes, the PR should state which layer is affected:

- Frontend or model import
- Relay IR
- Type or shape inference
- Relay passes
- TIR IR
- TIR passes
- Scheduling
- Code generation
- Runtime
- Device or target handling
- Profiling or diagnostics
- Build, CI, or developer tooling

## Validation Baseline

Use the narrowest checks that cover the issue. When the existing CMake CPU-only
toolchain is enough, prefer:

```powershell
cmake --preset dev-ninja-cpu
cmake --build --preset dev-ninja-cpu
cmake --build out/build/dev-ninja-cpu --target run_pass_pipeline_test
cmake --build out/build/dev-ninja-cpu --target run_profile_bundle_test
```

If a change touches CUDA, LLVM, code generation, device placement, or distributed
runtime behavior, add the relevant target-specific build and smoke test commands
to the PR.

## Review Rules

A PR is ready to merge when:

- The linked issue's acceptance criteria are satisfied.
- Required tests pass or any skipped checks are explicitly justified.
- The diff only contains changes needed for the issue.
- New public behavior is documented.
- Follow-up work is captured in separate issues instead of hidden in comments.

Do not merge PRs that combine infrastructure changes with operator semantics,
pass behavior, runtime behavior, or large documentation rewrites unless the issue
explicitly requires that combination.

## Agent Rules

When an AI coding agent works in this repository:

- Start from the issue and restate the implementation target.
- Work on a branch, not directly on `main`, unless the user explicitly asks for
  a local-only experiment.
- Before editing files, identify the files and behavior expected to change.
- After editing, run the relevant validation commands.
- Open or prepare a PR instead of pushing directly to `main`.
- Report exact commands run and whether they passed.
