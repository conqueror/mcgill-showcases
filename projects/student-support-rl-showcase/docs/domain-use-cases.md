# Domain Use Cases

## Primary domain

This showcase uses a synthetic student-support intervention policy. Each week, the policy chooses one action:

- `0`: no intervention
- `1`: send resource email
- `2`: recommend TA session
- `3`: escalate to advisor meeting

The state summarizes:

- week number,
- engagement score,
- assignment completion score,
- workload pressure,
- risk level,
- prior intervention count.

## Why this domain works

- It is familiar to the intended students.
- It is safe because it uses synthetic data only.
- It makes delayed effects and intervention cost easy to discuss.
- It highlights why governance matters even when the reward looks strong.

## Transfer ideas

Students should transfer the design pattern to a different domain, such as:

- support ticket prioritization,
- research workflow assistance,
- marketing journey intervention,
- supplier follow-up sequencing,
- learning-content recommendation.

They should not copy the same state, actions, or reward structure.
