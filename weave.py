import argparse
import os
from datetime import datetime

from utils import from_checkpoint, instantiate_roles, write_story_to_file

# Modal integration setup
try:
    import modal

    app = modal.App("weaver")

    # Create Modal image with all dependencies
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install_from_requirements("requirements.txt")
    )

except ImportError:
    modal = None
    app = None


def setup_args_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--max_iterations",
        type=int,
        required=True,
        help="Maximum number of iterations to use for the story",
    )
    parser.add_argument(
        "--multichar",
        action="store_true",
        help="Enable multi-character mode",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Start from a checkpoint",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        default=0.7,
        type=float,
        help="Temperature for the LLM model",
    )
    parser.add_argument(
        "-c",
        "--completion_tokens",
        default=2048,
        type=int,
        help="Maximum number of tokens per request",
    )
    return parser


def get_user_inputs(args) -> tuple[str, str | None]:
    """Get LLM and initial prompt from user."""
    if args.local:
        options = "Any HuggingFace model will do. Available options can be found at https://huggingface.co/models?pipeline_tag=text-generation."
    else:
        options = "Any model available on Groq will do. Available options can be found at https://console.groq.com/docs/models."
    llm = input(f">>> LLM to use ({options}): ")
    human_input = input(">>> Insert initial prompt: ") if not args.checkpoint else None
    return llm, human_input


def main(
    llm: str,
    human_input: str | None,
    local: bool,
    max_iterations: int,
    multichar: bool,
    temperature: float,
    completion_tokens: int,
) -> None:
    print("\n[+] Instantiating the roles...")
    narrator, worldsim, character, editor = instantiate_roles(
        llm, local, temperature, completion_tokens
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


# Modal remote function
if app is not None:

    @app.function(image=image, gpu="any", timeout=3600)
    def modal_main(
        llm: str,
        human_input: str | None,
        local: bool,
        max_iterations: int,
        multichar: bool,
        temperature: float,
        completion_tokens: int,
    ) -> tuple[str, str]:
        """Modal remote function that executes the main logic on Modal infrastructure."""
        import glob

        # Execute main logic - this will create files in the container
        main(llm, human_input, local, max_iterations, multichar, temperature, completion_tokens)

        # Find and read the generated story file
        story_pattern = os.path.join("stories", llm, "multichar" if multichar else "", "story_*.txt")
        story_files = glob.glob(story_pattern)

        if story_files:
            latest_story = max(story_files, key=os.path.getctime)
            with open(latest_story, "r") as f:
                story_content = f.read()
            # Return relative path for local recreation
            story_relpath = os.path.relpath(latest_story)
            return story_relpath, story_content
        else:
            raise FileNotFoundError("Story file was not generated")

    @app.local_entrypoint()
    def modal_cli():
        """Modal entry point for CLI argument parsing and user input."""
        parser = setup_args_parser()
        args = parser.parse_args()
        llm, human_input = get_user_inputs(args)

        # Execute main logic on Modal and get the story back
        story_path, story_content = modal_main.remote(
            llm,
            human_input,
            args.local,
            args.max_iterations,
            args.multichar,
            args.temperature,
            args.completion_tokens,
        )

        # Write story to local filesystem
        os.makedirs(os.path.dirname(story_path), exist_ok=True)
        with open(story_path, "w") as f:
            f.write(story_content)

        print(f"[+] Story saved to {story_path}")
        print("[+] All done!")


if __name__ == "__main__":
    parser = setup_args_parser()
    args = parser.parse_args()
    llm, human_input = get_user_inputs(args)

    main(
        llm,
        human_input,
        args.local,
        args.max_iterations,
        args.multichar,
        args.temperature,
        args.completion_tokens,
    )

    print("[+] All done!")
