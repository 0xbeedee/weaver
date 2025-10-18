import glob
import os


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
