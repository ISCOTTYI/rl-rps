from gymnasium.envs.registration import register

register(
    id="rps_game/GridWorld-v0",
    entry_point="rps_game.envs:GridWorldEnv",
)
