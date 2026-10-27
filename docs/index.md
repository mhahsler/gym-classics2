# Classic reinforcement learning, made inspectable

`gym_classics2` is a teaching package written by [Michael Hahsler](https://michael.hahsler.net/) in Python
covering finite Markov decision processes (MDPs), textbook reinforcement-learning algorithms,
and visualization tools.
The package is inspired by Brett Daley's gym-classics package and 
focuses on gridworld examples implemented as environments using the 
standard [Gymnasium API](https://gymnasium.farama.org/index.html). 
In addition to Gymnasium's sample access, `gym_classics2` environment also provide full model access 
for planning algorithms.

## Where to begin

- Follow [Getting started](getting-started.md) to install the package and run an
  environment.
- Browse [Environments](environments/overview.md) to choose a task.
- Read [Model access](environments/model-access.md) before using value or policy iteration.
- Use [Choosing an algorithm](algorithms/overview.md) to check an algorithm's
  requirements and outputs.
- Consult the [API reference](api/registration.md) for exact signatures.

## Design goals and teaching material

The package favors simple code, and inspectable intermediate results over framework abstractions. 
The implementations are intended for experiments, demonstrations, and coursework rather than
large-scale reinforcement-learning workloads.

The code maintains a close correspondence with the pseudocode in Sutton and Barto's 
textbook _Reinforcement Learning: An Introduction._ Teaching material for an RL course 
with `gym_classics2` (slides, examples, exercises) is available at
[Reinforcement Learning: Lecture Material, Simple Python Code Examples and Assignments.](https://michael.hahsler.net/Introduction_to_Reinforcement_Learning/)
