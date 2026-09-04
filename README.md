# model-forensics-after-dark

Model forensics asks *why* a model behaves concerningly (misalignment vs. benign confusion). This
mini-project studies a complementary question: **what governs an agent's behavior after it has
already identified and rejected a shortcut in its reasoning?** The goal is an earlier, empirically
validated intervention point for long-horizon agents that drift from a constraint they evidently
understand, and a way to detect and repair that drift on unseen models and tasks before a
concerning action becomes irreversible.

## Key takeaways

- **Reasoning-level refusal is not a stable aligned state.** "I should not do this" in the thinking
  tokens does not imply correct behavior; conceptual refusal of a shortcut is metastable.
- **Gaming risk grows with accumulated interaction history.** More context after a refusal
  substantially raises the chance the agent adopts the shortcut anyway.
- **Decodability is not prospective behavioral information.** Activation probes and J-lens/R-lens
  readouts decode task content but do not improve risk estimation or reliably rank sibling branches.
- **Constraint reminders act as behavioral control.** Re-injecting the constraint as text reduces
  gaming at 20 and 40 agent-steps (significant at 40), mostly via honest stopping rather than
  completed tasks.

## Setup

A coding agent (Meta Muse Glimmer 30B) is told to fix a repo's type errors without weakening,
bypassing, or suppressing the checker. It identifies a shortcut, rejects gaming in principle, and
later adopts it anyway, typically rationalizing that it is "not really" a blanket suppression.

A state-restorable **branching harness** freezes the conversation checkpoint and repository snapshot
to sample many sibling continuations, turning one observed trajectory into a distribution over
future behavior. **GLM-5.3** serves as judge over overlapping transcript windows
(`context_rot/annotation/`).

## Proxy tasks

- **Monitor.** *Velocity prediction:* freeze prefix and repo state at step *s*, sample
  continuations, estimate the probability of eventual gaming. *Sibling ranking:* from one state,
  test whether a score computed after *N* reasoning tokens separates gaming from non-gaming siblings.
- **Control.** Test whether an intervention reduces subsequent gaming while preserving helpfulness.
