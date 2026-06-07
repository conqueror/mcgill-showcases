"""Learning Agents Showcase — learn an agent's decision policy with RL/DRL.

A self-contained teaching package: it models an assistant's control loop as an RL environment and
learns the *orchestration policy* (routing, tool choice, escalation, stop) rather than retraining an
LLM. It vendors its own RL algorithm library and its own judge-rubric reward, so it has no
dependency on any other showcase.

The package is built in phases (see the repo plan): the agent environment and judge-rubric reward,
a vendored RL library (bandit, Q-learning, SARSA, dynamic programming, policy gradient, evaluation,
reporting), Lane-1 learning + governance, and optional lanes for offline RL, the OpenAI Agents SDK
bridge, RLHF/DPO/GRPO concepts, and multi-agent RL.
"""

__version__ = "0.1.0"
