from gymnasium.envs.registration import register

register(
    id="synthesis/GrammarSynthesisEnv-v0",
    entry_point="synthesis.envs:GrammarSynthesisEnv",
    max_episode_steps=200
)