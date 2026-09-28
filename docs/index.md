# Classic reinforcement learning, made inspectable

`gym_classics2` is a teaching package written in Python covering finite Markov decision
processes (MDPs), textbook reinforcement-learning algorithms, and visualization tools.
The package focuses on gridworld examples which are implemented as environments using the 
standard [Gymnasium API](https://gymnasium.farama.org/index.html) 
and also provide full model access for planning algorithms.

## Where to begin

- Follow [Getting started](getting-started.md) to install the package and run an
  environment.
- Browse [Environments](environments/overview.md) to choose a task.
- Read [Model access](environments/model-access.md) before using value or policy iteration.
- Use [Choosing an algorithm](algorithms/overview.md) to check an algorithm's
  requirements and outputs.
- Consult the [API reference](api/registration.md) for exact signatures.

## Design goals

The implementations favor simple code, correspondence with Sutton and Barto's pseudocode
and inspectable intermediate results over framework abstractions. They are
intended for experiments, demonstrations, and coursework rather than
large-scale reinforcement-learning workloads.
