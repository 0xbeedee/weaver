# weaver

weaver is a project exploring the possibility of creating stories automatically, by using an entirely artificial LLM-based end-to-end pipeline[^1].

The human enters the loop in the beginning, specifying the initial conditions, and then the rest is _woven_ together in an automated fashion.

## Running the code

The code uses `python 3.13`, and relies on [groq](https://groq.com)[^2] for LLM inference. To run it, do the following:

1. (Optional) Set up a virtual environment;

2. Install the project in editable mode (`pip install -e .`);

3. Put your API key in the `groq.key` file;

4. Run the `weave.py` script (`python weave.py`).

## Structure

`roles/` contains the classes corresponding to the various roles (character, editor, narrator and worldsim);

`system_prompts/` contains the system prompts for the various roles.

`logs/` contains the logs of the runs, it also represents the memory of the system.

[^1]: quite the mouthful...
[^2]: this is a rather stringent limitation, but it fits the PoC nature of the project.
