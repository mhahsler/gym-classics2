# Classic reinforcement learning, made inspectable

`gym_classics2` is a teaching package containing finite Markov decision
processes, textbook reinforcement-learning algorithms, and visualization tools.
Its environments implement the standard Gymnasium API and also expose their
transition models for planning algorithms.

## Where to begin

- Follow [Getting started](getting-started.md) to install the package and run an
  environment.
- Browse [Environments](environments/overview.md) to choose a task.
- Read [Model access](environments/model-access.md) before using value or policy iteration.
- Use [Choosing an algorithm](algorithms/overview.md) to check an algorithm's
  requirements and outputs.
- Consult the [API reference](api/registration.md) for exact signatures.

## Design goals

The implementations favor correspondence with Sutton and Barto's pseudocode
and inspectable intermediate results over framework abstractions. They are
intended for experiments, demonstrations, and coursework rather than
large-scale reinforcement-learning workloads.
