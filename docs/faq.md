# FAQ

## Do I need all projects to run?
No. Start with one project.

## Do I need Kaggle credentials?
Only for `causalml-kaggle-showcase`.

## Why are artifact folders mostly empty?
Generated artifacts are excluded from git to keep the repo lightweight. Run project scripts to regenerate them.

## What does `make check` do at root?
It runs lint, type checks, and tests across all projects.

## Why are there project-level Makefiles and a root Makefile?
- Root Makefile: orchestrates all projects.
- Project Makefile: detailed workflows specific to that project.
