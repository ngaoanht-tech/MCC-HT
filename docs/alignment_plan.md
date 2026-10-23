# Paper alignment with retained joint projection

User scope: align existing code to Manuscript_clean_R; preserve joint projection and all three call paths per user steering. Preserve original archive and audit source.

1. Regression tests: quadratic smoothness, shared alignment weight and unnormalized terminal MSE, unclamped raw generation, feasibility-based 2/3 outer updates, terminal-only FR and reservoir-year TPR, timer averaging.
2. Modify losses and raw capacity head; align internal training target and effective power units.
3. Replace flow-difference stopping with physical constraint and terminal-level checks using reconstructed cascade storage. Keep joint projection after outer loop. Record iteration count and feasibility separately.
4. Export initial pre-ICR and final post-joint stages; add paper metric export and five-repeat timing entry. Full test-year configuration separated from disjoint small-data example.
5. Execute unit tests using installed D:/Python Python with PyTorch, existing relevant regressions and available-input smoke inference if artifacts permit. No invented checkpoint/data/results.
6. Package separate archive with change log and exact verification limits.
