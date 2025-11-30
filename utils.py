import glob
import os

from roles import Character, Editor, Narrator, WorldSim


def instantiate_roles(
    llm: str, local: bool, temperature: float, completion_tokens: int
):
    # common kwargs
    gen_kwargs = {
        "temperature": temperature,
        # the arguments below are best left as default values
        "top_p": 1,
        "stream": False,
        "stop": None,
    }
    if local:  # huggingface
        gen_kwargs["max_new_tokens"] = completion_tokens
        gen_kwargs["do_sample"] = True if temperature > 0 else False
        del gen_kwargs["stream"]
    else:  # groq
        gen_kwargs["max_completion_tokens"] = completion_tokens

    narrator = Narrator(llm=llm, local=local, gen_kwargs=gen_kwargs)
    worldsim = WorldSim(llm=llm, local=local, gen_kwargs=gen_kwargs)
    character = Character(llm=llm, local=local, gen_kwargs=gen_kwargs)
    editor = Editor(llm=llm, local=local, gen_kwargs=gen_kwargs)

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
