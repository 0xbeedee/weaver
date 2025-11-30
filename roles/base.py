import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from groq import Groq
from transformers import pipeline


class BaseRole:
    """A base class for all the roles.

    It centralises LLM generation and handles the roles' memory. It also automatically logs all interactions to a timestamped file in a 'logs' directory.
    """

    def __init__(
        self, role: str, llm: str, local: bool, gen_kwargs: Dict[str, Any] = {}
    ) -> None:
        """
        Initializes the role and creates a timestamped log directory.

        Args:
            role: The role of the model.
            llm: The name of the model to use for the text generation.
        """
        if local:  # huggingface
            self.pipe = pipeline("text-generation", model=llm)
        else:  # groq
            with open("groq.key", "r") as f:
                api_key = f.read().strip()
            self.client = Groq(api_key=api_key)

        self.local = local
        self.gen_kwargs = gen_kwargs
        self.model = llm

        self.role: str = role
        self.system_prompt: str = Path(f"system_prompts/{role}.txt").read_text()

        # a simplified memory, i.e., a list of past prompts
        self.memory: List[str] = []

        # logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp for uniqueness
        log_dir = Path("logs") / timestamp
        log_dir.mkdir(exist_ok=True, parents=True)
        self.log_file_path = log_dir / f"{self.role}.log"

        # use the roles to disambiguate loggers
        self.logger = logging.getLogger(self.role)
        self.logger.setLevel(logging.INFO)

        # avoid adding duplicate handlers for multiple instances
        if not self.logger.hasHandlers():
            handler = logging.FileHandler(self.log_file_path)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s: %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(
            f"BaseRole initialized for role: '{self.role}'. Logging to: {self.log_file_path}"
        )

    def generate(
        self,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        save_to_memory: bool = True,
    ) -> str:
        """
        Generates text based on the given prompt using its role and system prompt.
        Logs interactions to the automatically generated log file.

        Args:
            prompt: The input prompt for the language model.
            save_to_memory: Whether to save the input prompt and output to memory.

        Returns:
            The generated text.
        """
        self.logger.info(
            f"Role '{self.role}' - generating with prompt: '{user_prompt}...'"
        )

        messages = []
        if system_prompt:
            # if the caller provides a system prompt, use it instead of the one found in the txt files
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self.local:
            chat_completion = self.pipe(messages, **self.gen_kwargs)
            generated_text = self._clean_text_from_pipeline(chat_completion)
        else:
            chat_completion = self.client.chat.completions.create(
                messages=messages, model=self.model, **self.gen_kwargs
            )
            generated_text = chat_completion.choices[0].message.content

        if generated_text is None:
            raise ValueError("Generated text is None")

        self.logger.info(f"Role '{self.role}' - generated text: '{generated_text}...'")

        if save_to_memory:
            # input_entry = f"Input Prompt: {user_prompt}"
            output_entry = f"Generated Output: {generated_text}"

            # self.memory.append(input_entry)
            self.memory.append(output_entry)

            self.logger.debug(
                f"Saved memory for role '{self.role}'. Memory size: {len(self.memory)} entries."
            )

        return generated_text

    def get_memory(self) -> List[str]:
        """
        Retrieves the entire memory of the role model.

        Returns:
            A list of strings representing the memory log.
        """
        return self.memory.copy()  # return a copy to prevent external modification

    def clear_memory(self) -> None:
        """
        Clears the role model's memory.
        The system prompt, if set, will be re-added and re-logged.
        """
        self.memory = []
        self.logger.info(f"Memory cleared for role '{self.role}'.")

    def _clean_text_from_pipeline(
        self, chat_completion: list[dict[str, list[dict]]]
    ) -> str:
        """Clean the text from the transformers' pipeline."""
        # see https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.TextGenerationPipeline for output format
        thinking_out = chat_completion[0]["generated_text"][-1]["content"]
        # remove thinking outputs
        return re.sub(r"<think>.*?</think>", "", thinking_out, flags=re.DOTALL).strip()
