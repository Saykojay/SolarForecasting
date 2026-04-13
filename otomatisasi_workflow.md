# otomatisasi update readme,tutorial,github

This document defines the specialized agent workflow for the **Solar Forecasting Pipeline**. Whenever the USER requests a shorthand command, the agent must follow these procedures.

## Shorthand Commands

| Command | Action |
| :--- | :--- |
| `solar-sync` | Stages all changes, commits with a standard prefix (`docs:`, `feat:`, `fix:`), and pushes to `origin main`. |
| `solar-status` | Runs `git status`, checks the `models/` folder for the newest weights, and summarizes current pipeline readiness. |
| `solar-run` | Executes the batch training script `scripts/run_batch_train.py` and reports results. |
| `solar-doc-update` | Scans `src/` and `scripts/` for new logic or parameters and updates `README.md` and `TUTORIAL.md` accordingly. |

## Workflow Protocols

1. **Prioritize Physics**: When documentation is updated, always emphasize **Algorithm 1 (Physics-based Cleaning)** as the core differentiator/novelty.
2. **Context Awareness**: Always reference the Stage-to-Stage integration (Forecasting → HOMER Pro → Aspen Plus) to maintain research consistency.
3. **Commit Messages**: Use professional conventional commits.
   - Example: `docs: update tutorial for TMY synthesis`
   - Example: `feat: add Open-Meteo acquisition script`

---
*Created by Antigravity Solar Agent Workflow v1*
