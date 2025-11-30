from typing import Any, Dict

from .base import BaseRole


class WorldSim(BaseRole):
    """
    A specialized language model for simulating world events.
    """

    def __init__(
        self,
        llm: str = "moonshotai/kimi-k2-instruct-0905",
        local: bool = False,
        gen_kwargs: Dict[str, Any] = {},
    ):
        super().__init__(role="worldsim", llm=llm, local=local, gen_kwargs=gen_kwargs)

    def simulate_world_event(self, current_situation_prompt: str) -> str:
        """
        Simulates a world event based on the current situation.
        The generated output is saved to this model's memory.
        """
        task_specific_prompt = f"As the world simulator, describe what happens next given the situation: '{current_situation_prompt}'"
        return self.generate(user_prompt=task_specific_prompt)
