from gymnasium.envs.registration import register

register(
    id="rps_game/RPS-v0",
    entry_point="rps_game.envs:RPSEnv",
)
