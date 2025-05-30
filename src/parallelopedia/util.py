# =============================================================================
# Imports
# =============================================================================
import importlib.util
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os.path import abspath, join, normpath
from typing import List

# =============================================================================
# Aliases
# =============================================================================
basename = os.path.basename

# =============================================================================
# Globals
# =============================================================================


# =============================================================================
# Context Managers
# =============================================================================
class ElapsedTimer:
    """
    Context manager and reusable timer to measure elapsed time.

    Example:
        timer = elapsed_timer()
        with timer:
            do_something()
        print(f'Elapsed: {timer.elapsed:.3f}')

        # Re-enterable:
        with timer:
            do_something_else()
        print(f'Elapsed: {timer.elapsed:.3f}')
    """

    def __init__(self):
        self.start = None
        self._elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._elapsed = time.perf_counter() - self.start

    @property
    def elapsed(self):
        """
        Return the elapsed time for the most recent context.
        """
        if self._elapsed is None:
            raise ValueError("Timer has not been used in a context yet.")
        return self._elapsed

# =============================================================================
# Helpers
# =============================================================================


def join_path(*args):
    return abspath(normpath(join(*args)))


def get_class_from_string(class_name: str,
                          enforce_flake8: bool = False) -> type:
    """
    Obtains an instance of a class object from a string representation of the
    class name, which may include the module name, e.g. `spam.eggs.Bacon`.

    Args:

        class_name (str): Supplies the name of the class.

        enforce_flake8 (bool): Optionally supplies a boolean that, if True,
            runs flake8 on the imported module and raises an exception if any
            issues are detected.

    Returns:
        type: Returns the class object.

    Raises:
        Exception: If flake8 detects any issues in the imported module.
    """
    timer = ElapsedTimer()
    with timer:
        parts = class_name.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        module = None

        if not module_name:
            # Attempt to resolve the class name directly from globals.
            module = globals()[class_name]

        elif enforce_flake8:
            # Find the module's spec to get the file path without importing
            # it first.  This allows us to run flake8 and fail-fast if any
            # issues are detected.
            spec = importlib.util.find_spec(module_name)
            if not spec or not spec.origin:
                raise ModuleNotFoundError(
                    f"Module '{module_name}' could not be located."
                )

            module_file = spec.origin

            result = subprocess.run(
                ["flake8", module_file],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(
                    f"Flake8 issues found in {module_file}:\n"
                    f"{result.stdout}{result.stderr}"
                )

        if not module:
            module = __import__(module_name)
            # Traverse the module hierarchy to get the class object.
            for comp in parts[1:]:
                module = getattr(module, comp)

    logging.info(f'Loaded {class_name} in {timer.elapsed:.4f} seconds.')
    if hasattr(module, 'init_once'):
        logging.info(f'Calling {class_name}.init_once()...')
        module.init_once()
    return module


def get_classes_from_strings_parallel(class_names: List[str]) -> List[type]:
    """
    Obtains a list of class objects from a list of string representations of
    class names, which may include the module name, e.g. `spam.eggs.Bacon`.

    Args:

        class_names (List[str]): Supplies a list of class names.

    Returns:

        List[type]: Returns a list of class objects.

    """
    max_workers = min(os.cpu_count(), len(class_names))
    results = []
    errors = []
    logging.info(f'Loading {len(class_names)} classes in parallel...')
    logging.info(f'Max workers: {max_workers}')
    logging.info(f'Class names: {class_names}')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(get_class_from_string, class_name): class_name
            for class_name in class_names
        }
        for future in as_completed(futures):
            class_name = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f'Error loading {class_name}: {e}')
                errors.append(e)
    if errors:
        raise Exception(f'Errors occurred while loading classes: {errors}')
    return results

@dataclass
class HuggingFaceModel:
    name: str
    config: dict
    safetensors: "safetensors.safe_open"
    tokenizer: dict
    tokenizer_config: dict
    vocab: dict

def get_huggingface_model(model_name: str) -> HuggingFaceModel:
    """
    Returns a Hugging Face model object for the given model name.

    Args:

        model_name (str): Supplies the name of the Hugging Face model.  This
            should be in the format of `namespace/model`, e.g. for GPT2 XL:
            `openai-community/gpt2-xl`.  This will be expanded out to the
            following directory:
                `~/.cache/huggingface/hub/models--openai-community--gpt2-xl`

    Returns:

        HuggingFaceModel: Returns a HuggingFaceModel object containing the
            model name, configuration, and SafeTensors object.
    """
    base = os.path.expanduser('~/.cache/huggingface/hub/models--')
    (namespace, model) = model_name.split('/')
    base_path = f'{base}{namespace}--{model}'
    ref_path = f'{base_path}/refs/main'
    with open(ref_path, 'r') as f:
        ref = f.read().strip()
    snapshots_dir = f'{base_path}/snapshots/{ref}'
    safetensors_path = f'{snapshots_dir}/model.safetensors'
    import safetensors
    timer = ElapsedTimer()
    logging.debug(f'About to load safetensors from {safetensors_path}...')
    with timer:
        st = safetensors.safe_open(
            safetensors_path,
            framework="pt",
            device="cpu",
        )
    msg = (
        f'Loaded safetensors from {safetensors_path} '
        f'in {timer.elapsed:.4f} seconds.'
    )
    logging.info(msg)

    config_path = f'{snapshots_dir}/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    tokenizer_path = f'{snapshots_dir}/tokenizer.json'
    with open(tokenizer_path, 'r') as f:
        tokenizer = json.load(f)

    tokenizer_config_path = f'{snapshots_dir}/tokenizer_config.json'
    with open(tokenizer_config_path, 'r') as f:
        tokenizer_config = json.load(f)

    vocab_path = f'{snapshots_dir}/vocab.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    return HuggingFaceModel(
        model_name,
        config,
        st,
        tokenizer,
        tokenizer_config,
        vocab,
    )

# vim:set ts=8 sw=4 sts=4 tw=78 et:                                           #
