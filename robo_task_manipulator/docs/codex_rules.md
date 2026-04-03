Project rules for RoboTaskManipulator

Scope
- Build this feature inside `robo_task_manipulator/` only unless explicitly told otherwise.
- Do not modify unrelated SaaS code.
- If you must rename, move, or delete files, update imports and docs accordingly.

Product goal
- Build a practical v1 product that takes photos or videos as input and outputs:
  - ordered task steps
  - conservative symbolic labels
  - useful context/failure tags
  - simulation-ready structured output for Isaac Sim 5.1 / Franka Panda
  - evaluation-friendly outputs
- Prioritize a visible, working end-to-end product.

Architecture priorities
- Prefer a hybrid pipeline over a single-model design if that improves quality.
- Do NOT assume pi0 must be the only model used.
- Semantic understanding should be VLM-first.
- Action/policy backends are optional and should be used only where they add value.
- If an action backend is used, it can be pi0, OpenVLA, or none.
- The rest of the pipeline must still work even if no action backend is enabled.

Code organization
- Organize the codebase by responsibility so it is obvious where to edit each part.
- Each major pipeline stage should have one clear home.
- Preferred responsibilities include:
  - ingestion
  - segmentation
  - understanding
  - action backend
  - context/failure tagging
  - graph/sequencing
  - simulation export
  - evaluation
  - final export
  - shared schemas/utils
- Avoid scattering related logic across multiple folders/files.
- Avoid generic filenames when a specific one is clearer.

Cleanup
- Remove unused, duplicate, obsolete, placeholder, experimental, or dead files that are not part of the current product.
- Do not keep old files “just in case” if they are no longer used.
- If two files do nearly the same thing, merge or remove one.
- Keep the final tree lean and understandable.

Implementation style
- Prefer practical, readable Python.
- Avoid overengineering.
- Prefer simple deterministic logic for v1 where appropriate.
- Use pretrained inference; do not build a heavy training pipeline.
- Keep functions focused.
- Add docstrings to public classes/functions.
- Add useful logging.
- Make errors actionable.

Quality rules
- Be conservative when uncertain.
- Prefer `unknown` over a guessed label.
- Do not fabricate certainty.
- Do not claim physics certainty from image/video alone.
- Make assumptions explicit in comments and docs.
- Preserve raw model outputs where useful for debugging.

Evaluation
- Always keep evaluation in mind.
- Outputs should be easy to compare against a small benchmark/golden set.
- Prefer outputs that are easy to validate over outputs that are only theoretically richer.

Non-goals
- No UI.
- No heavy training system.
- No unnecessary platform abstraction.
- No hard dependency on robot state as input.
- No hard lock to a single model if a mixed approach is better.