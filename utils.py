import glob
import os

from roles import Character, Editor, Narrator, WorldSim


def instantiate_roles(llm: str, temperature: float, completion_tokens: int):
    global_kwargs = {
        "temperature": temperature,
        "max_completion_tokens": completion_tokens,
        # the arguments below are best left as default values
        "top_p": 1,
        "stream": False,
        "stop": None,
    }
    narrator = Narrator(llm=llm, groq_kwargs=global_kwargs)
    worldsim = WorldSim(llm=llm, groq_kwargs=global_kwargs)
    character = Character(llm=llm, groq_kwargs=global_kwargs)
    editor = Editor(llm=llm, groq_kwargs=global_kwargs)

    return narrator, worldsim, character, editor


def write_story_to_file(story: str, filename: str) -> None:
    """Write the completed story to a file."""
    with open(filename, "w") as file:
        file.write(story)


def from_checkpoint(llm, multichar):
    """Load the text from the most recent file produced by the LLM specified (to use it as a starting point for the new story)."""
    path = f"stories/{llm}/multichar/*" if multichar else f"stories/{llm}/*"
    files = glob.glob(path)
    if not files:
        raise FileNotFoundError(f"[!] No files found for {path}!")
    most_recent_file = max(files, key=os.path.getctime)
    with open(most_recent_file, "r") as file:
        return file.read()
