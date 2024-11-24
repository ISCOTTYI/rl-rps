# 2D Rock-Paper-Scissors as Gymnasium Environment
Some simple examples of Gymnasium environments and wrappers.
For some explanations of these examples, see the [Gymnasium documentation](https://gymnasium.farama.org).

### Game
Two players play against each other. Each player gets initially some rocks,
papers and scissors. The players take turns. In each round one player moves one
of its pieces up, down, left or right one step. Pieces generally block each
other, except opposing pieces that can engage in a rock-paper-scissors (RPS) fight
(i.e. different types of pieces). In a RPS fight, the loosing piece is annihilated.
The first player to lose all its pieces, loses the game.

### Wrappers
This repository hosts the examples that are shown [on wrapper documentation](https://gymnasium.farama.org/api/wrappers/).
- `ClipReward`: A `RewardWrapper` that clips immediate rewards to a valid range
- `DiscreteActions`: An `ActionWrapper` that restricts the action space to a finite subset
- `RelativePosition`: An `ObservationWrapper` that computes the relative position between an agent and a target
- `ReacherRewardWrapper`: Allow us to weight the reward terms for the reacher environment

## Installation

To install your new environment, run the following commands:

```{shell}
cd rps_game
pip install -e .
```

