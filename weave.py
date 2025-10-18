import os
import argparse
from datetime import datetime

from roles import Narrator, WorldSim, Character, Editor
from utils import write_story_to_file, from_checkpoint


def main(
    human_input: str | None,
    max_iterations: int,
    multichar: bool,
    checkpoint: bool,
    llm: str,
    temperature: float,
    completion_tokens: int,
) -> None:
    # TODO should make the whole thing more configurable, hydra is probably the way...

    print("[+] Instantiating the roles...")
    # use one set of kwargs for simplicity
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

    print("[+] Initialising the narrator...")
    init_prompt = human_input if human_input else from_checkpoint(llm, multichar)
    prompt = narrator.edit_input(init_prompt)
    for i in range(max_iterations):
        print(f"[+] Iteration {i + 1}:")
        print("\t[+] WorldSim - Simulating...")
        sim_out = worldsim.simulate_world_event(prompt)
        print("\t[+] Narrator - Editing simulation output...")
        prompt = narrator.edit_simulation_output(sim_out, prompt)
        print("\t[+] Character - Deciding action...")
        action = character.decide_action(prompt, multichar)
        print("\t[+] Narrator - Narrating action...")
        prompt = narrator.narrate_action(action, prompt)

    print("[+] Finalising the story...")
    memory_dict = {
        "narrator": narrator.get_memory(),
        "worldsim": worldsim.get_memory(),
        "character": character.get_memory(),
    }
    story = editor.compile_story(memory_dict)

    # don't add checkpointing to the path to make the weaving more seamless
    base_path = os.path.join("stories", llm)
    if multichar:
        base_path = os.path.join(base_path, "multichar")
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # for uniqueness
    story_fname = os.path.join(base_path, f"story_{timestamp}.txt")

    print(f"[+] Saving the story to {story_fname}...")
    write_story_to_file(story, story_fname)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-m",
        "--max_iterations",
        type=int,
        required=True,
        help="Maximum number of iterations to use for the story",
    )
    args.add_argument(
        "--multichar",
        action="store_true",
        help="Enable multi-character mode",
    )
    args.add_argument(
        "--checkpoint",
        action="store_true",
        help="Start from a checkpoint",
    )

    # llm-specific arguments
    args.add_argument(
        "-l",
        "--llm",
        default="moonshotai/kimi-k2-instruct-0905",
        type=str,
        help="LLM model to use for the story (MUST BE IN THE LIST OF AVAILABLE MODELS ON Groq!)",
    )
    args.add_argument(
        "-t",
        "--temperature",
        default=0.7,
        type=float,
        help="Temperature for the LLM model",
    )
    args.add_argument(
        "-c",
        "--completion_tokens",
        default=2048,
        type=int,
        help="Maximum number of tokens per request",
    )

    args = args.parse_args()

    human_input = input(">>> Insert initial prompt: ") if not args.checkpoint else None
    main(
        human_input if human_input else None,
        args.max_iterations,
        args.multichar,
        args.checkpoint,
        args.llm,
        args.temperature,
        args.completion_tokens,
    )

    print("[+] All done!")
