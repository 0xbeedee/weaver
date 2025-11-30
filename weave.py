import argparse
import os
from datetime import datetime

from utils import from_checkpoint, instantiate_roles, write_story_to_file


def main(
    llm: str,
    human_input: str | None,
    max_iterations: int,
    multichar: bool,
    checkpoint: bool,
    temperature: float,
    completion_tokens: int,
) -> None:
    # TODO should make the whole thing more configurable, hydra is probably the way...

    print("\n[+] Instantiating the roles...")
    narrator, worldsim, character, editor = instantiate_roles(
        llm, temperature, completion_tokens
    )

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
    args.add_argument(
        "--local",
        action="store_true",
        help="Run locally",
    )

    # llm-specific arguments
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

    # having the user input the model (instead of using a CLI flag) makes for a cleaner interface
    if args.local:
        options = "Any HuggingFace model will do. Available options can be found at https://huggingface.co/models?pipeline_tag=text-generation."
    else:
        options = "Any model available on Groq will do. Available options can be found at https://console.groq.com/docs/models."
    llm = input(f">>> LLM to use ({options}): ")

    human_input = input(">>> Insert initial prompt: ") if not args.checkpoint else None
    main(
        llm,
        human_input if human_input else None,
        args.max_iterations,
        args.multichar,
        args.checkpoint,
        args.temperature,
        args.completion_tokens,
    )

    print("[+] All done!")
