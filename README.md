# weaver

weaver is a project exploring the possibility of creating stories automatically, by using an entirely artificial LLM-based end-to-end pipeline[^1].

The human enters the loop in the beginning, specifying the initial conditions, and then the rest is _woven_ together in an automated fashion.

## Running the code

The code uses `python 3.13`. 
It can be run by either relying on [groq](https://groq.com), or by using [HuggingFace](https://huggingface.co).

### Running with groq

To use groq, do the following:

1. (Optional) Set up a virtual environment;

2. Install the project in editable mode (`pip install -e .`);

3. Put your API key in the `groq.key` file;

4. Run the `weave.py` script (`python weave.py`).

### Running with HuggingFace

You can use HuggingFace to either run inference locally/on classical GPUaaS providers like [Lambda](https://lambda.ai), or remotely, using [Modal](https://modal.com).

In both cases, the first step remains the same:

1. (Optional) Set up a virtual environment;

#### Running locally/using a GPUaaS

2. Install the project in editable mode (`pip install -e .`);

3. Run the `weave.py` script (`python weave.py`).

#### Running with Modal

2. Set up Modal, following the [instructions](https://modal.com/docs/guide#getting-started) on their website.

3. Install the project in editable mode (`pip install -e .[modal]`);

4. Run the `weave.py` script using modal (`modal run weave.py`).

Modal execution accepts the same CLI arguments as standard Python execution (e.g., `modal run weave.py -m 5 --multichar`).

## Structure

`roles/` contains the classes corresponding to the various roles (character, editor, narrator and worldsim);

`system_prompts/` contains the system prompts for the various roles.

`logs/` contains the logs of the runs, it also represents the memory of the system.

[^1]: quite the mouthful...
