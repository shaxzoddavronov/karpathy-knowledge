# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning repository that reimplements Andrej Karpathy's "Neural Networks: Zero to Hero" material from scratch. Work happens in Jupyter notebooks, not `.py` modules — the code is exploratory and evolves cell by cell as concepts are built up. Treat notebooks as the primary artifact; do not refactor them into packages unless asked.

Current content: `building-micrograd/micrograd_from_scratch.ipynb` — a hand-built scalar autograd engine (micrograd).

## Environment

- Python 3.12 in a venv located **one level above the repo root**: `../.venv` (i.e. `C:\Users\acer\Desktop\shakhzod-scripts\.venv`). The repo itself is `karpathy-knowledge/`.
- Key packages: `numpy`, `matplotlib`, `graphviz` (Python binding), `ipykernel`/`jupyter`.
- **Graphviz binary requirement:** the `graphviz` Python package is only a wrapper — it shells out to the `dot` executable, which must be on `PATH`. The native binaries are vendored one level above the repo at `../graphviz-dist/Graphviz-12.2.1-win64/bin` (contains `dot.exe`). If `draw_dot(...)` raises an `ExecutableNotFound` error, that bin directory is missing from `PATH`.

Activate and run (PowerShell):
```powershell
..\.venv\Scripts\Activate.ps1
$env:PATH += ";$(Resolve-Path ..\graphviz-dist\Graphviz-12.2.1-win64\bin)"
jupyter notebook building-micrograd\micrograd_from_scratch.ipynb
```

There is no build, test, or lint setup — this is a notebook-driven learning repo.

## Architecture: the `Value` autograd engine

The core is a single `Value` class that wraps one scalar and records how it was produced, forming a computation graph that supports reverse-mode autodiff. Understanding the mechanism requires reading how these pieces interlock:

- **Graph construction:** every arithmetic dunder (`__add__`, `__mul__`, `tanh`, ...) returns a *new* `Value` whose `_prev` holds the operand nodes and `_op` records the operation. The graph is built implicitly as the forward expression is evaluated.
- **Local backward closures:** each operation attaches a `_backward` closure to its output node. The closure knows the local derivative rule for that op and pushes gradient into the operands' `.grad` using `out.grad` (the chain rule). `_backward` defaults to a no-op for leaf nodes.
- **Gradients** accumulate on `.grad` (initialized to `0.0`). Note: the current closures **assign** rather than `+=`, so they are not yet correct for nodes reused in multiple paths — this is an in-progress teaching artifact, not a finished library.
- **Visualization:** `trace(root)` walks `_prev` edges to collect nodes/edges; `draw_dot(root)` renders them with graphviz, showing each value's `label`, `data`, and `grad`. This is the primary debugging tool for the graph.

Cells frequently set `.grad` values and call `_backward()` manually in sequence to demonstrate backpropagation step by step, and compare against numerical gradients (finite differences, e.g. the `lol()` cell) to verify the analytic derivatives.
