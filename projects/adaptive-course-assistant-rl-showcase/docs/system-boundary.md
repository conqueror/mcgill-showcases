# System Boundary

This showcase draws a hard line between two responsibilities.

## Deterministic Assistant Responsibilities

- classify the request,
- retrieve an initial course resource,
- stay within the assistant's allowed teaching scope.

## Learned Policy Responsibilities

- decide whether to clarify,
- decide whether to retrieve more evidence,
- decide whether to hint or give a worked example,
- decide whether to slow down,
- decide whether to assign targeted practice,
- decide whether to escalate.

That split keeps the project honest.

If we let the policy generate the whole answer, the simulator turns into a vague story about "AI helping students." By keeping the content layer fixed and the intervention layer learned, we can actually inspect what the agent is optimizing.
