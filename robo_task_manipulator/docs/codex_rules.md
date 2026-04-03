Project rules for RoboTaskManipulator

- Build this feature inside robo_task_manipulator/ only unless explicitly told otherwise.
- Do not modify unrelated SaaS code.
- Keep model-specific logic isolated behind adapters.
- Use pi0 as the VLA backend, not SmolVLA.
- Prioritize 4a Task Understanding and 4b Action Layer first.
- Structure code for later extension to 4c, 4d, 4e, and 4f.
- Use strict typed schemas for annotations.
- Save normalized outputs and raw model outputs separately.
- Prefer simple, readable Python over over-engineered abstractions.
- Add TODO comments where hardware/model-specific assumptions are unclear.
- Do not add frontend/UI unless explicitly requested.
- Do not invent missing dataset fields silently; mark assumptions clearly.