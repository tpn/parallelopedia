# =============================================================================
# Imports
# =============================================================================
import glob
import logging
import os
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from os.path import abspath, join, normpath
from typing import List, Tuple

import datrie
import numpy as np

# =============================================================================
# Aliases
# =============================================================================
basename = os.path.basename

# =============================================================================
# Globals
# =============================================================================
ALLOWED = string.printable + string.punctuation

ORD_ALLOWED = set(map(ord, ALLOWED))

PARTITIONS = 127

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


def get_class_from_string(class_name: str) -> type:
    """
    Obtains an instance of a class object from a string representation of the
    class name, which may include the module name, e.g. `spam.eggs.Bacon`.

    Args:

        class_name (str): Supplies the name of the class.

    Returns:

        type: Returns the class object.

    """
    timer = ElapsedTimer()
    with timer:
        parts = class_name.split('.')
        module_name = '.'.join(parts[:-1])
        class_name = parts[-1]
        if module_name:
            module = __import__(module_name)
            for comp in parts[1:]:
                module = getattr(module, comp)
        else:
            module = globals()[class_name]
    logging.info(f'Loaded {class_name} in {timer.elapsed:.4f} seconds.')
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

def extract_trie(trie: datrie.Trie, chars: Tuple[str]) -> datrie.Trie:
    """
    Creates a new trie populated by all items in the original trie that are
    associated with the given characters.

    Args:

        trie (datrie.Trie): Supplies the source datrie.Trie from which items
                            will be extracted.

        chars (Tuple[str]): Supplies an iterable of characters that will be
                            used to extract items from the source trie and
                            added to the new trie.

    Returns:

        datrie.Trie: A new trie with the items associated with the given
        characters.
    """
    result = datrie.Trie(ALLOWED)
    for c in chars:
        items = trie.items(c)
        for key, value in items:
            if key in result:
                existing = result[key]
                for v in value:
                    if v not in existing:
                        existing.append(v)
                        existing.sort()
            else:
                result[key] = value
    return result


def extract_and_save(trie: datrie.Trie, chars: Tuple[str]):
    """
    Creates a new trie populated by all items in the original trie that are
    associated with the given characters.

    Args:

        trie (datrie.Trie): Supplies the source datrie.Trie from which items
            will be extracted.

        chars (Tuple[str]): Supplies an iterable of characters that will be
            used to extract items from the source trie and added to the new
            trie.  This could be a tuple of one item of a single character,
            multiple items of single characters, or combinations of single and
            multiple characters.  The trie will be queried for matches via
            `trie.items(c)` for each character `c` in `chars`.

    Returns:

        None
    """
    msg_prefix = f'[{threading.get_native_id()}]'
    result = datrie.Trie(ALLOWED)
    start = time.perf_counter()
    for c in chars:
        items = trie.items(c)
        for key, value in items:
            if key in result:
                existing = result[key]
                for v in value:
                    if v not in existing:
                        existing.append(v)
                        existing.sort()
            else:
                result[key] = value
    num_results = len(result)
    end = time.perf_counter()
    elapsed = end - start
    if num_results == 0:
        print(
            f'{msg_prefix} No items extracted for chr({ord(c)}) '
            f'(elapsed: {elapsed:.4f} seconds).'
        )
        return

    msg = (
        f'{msg_prefix} Extracted {num_results} items for chr({ord(c)}) '
        f'in {elapsed:.4f} seconds.'
    )
    print(msg)

    filename_prefix = 'wiki-' + '-'.join(str(ord(c)).zfill(3) for c in chars)
    filename = f'{filename_prefix}_{num_results}.trie'
    msg = f'{msg_prefix} Saving {num_results} items to {filename}...'
    start = time.perf_counter()
    result.save(filename)
    end = time.perf_counter()
    elapsed = end - start
    msg = (
        f'{msg_prefix} Saved {num_results} items to {filename} '
        f'in {elapsed:.4f} seconds.'
    )
    print(msg)


def extract_and_save_all_chars_parallel(
    trie: datrie.Trie, max_threads: int = 0
) -> None:
    """
    Extracts all items from the given trie for each character in the `ALLOWED`
    string and saves the results to a file named `wiki-XXX_Y.trie`, where `XXX`
    is the ordinal value of the character and `Y` is the number of items saved.

    Args:

        trie (datrie.Trie): Supplies the source datrie.Trie from which items
            will be extracted.

        max_threads (int): Supplies the maximum number of threads to use for
            parallel processing.  If <= zero, the number of threads will be
            set to the number of CPUs available on the system.

    Returns:

        None
    """

    if max_threads < 1:
        max_threads = os.cpu_count()

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(extract_and_save, trie, (c,)): c for c in ALLOWED
        }
        for future in as_completed(futures):
            c = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'[{threading.get_native_id()}] {c}: {e}')


def get_wiki_tries_in_dir(directory: str) -> List[str]:
    return sorted(glob.glob(f'{directory}/wiki-*.trie'))


def get_wiki_tries(directory: str) -> dict:
    paths = get_wiki_tries_in_dir(directory)
    result = {}
    for path in paths:
        base = basename(path).replace('wiki-', '').replace('.trie', '')
        parts = base.split('_')
        assert len(parts) == 2, f'Invalid filename: {base}'
        (ordinal, length) = parts
        char = chr(int(ordinal))
        result[char] = path
    return result


def load_trie(path: str) -> datrie.Trie:
    msg_prefix = f'[{threading.get_native_id()}]'
    start = time.perf_counter()
    print(f'{msg_prefix} Loading {path}...')
    trie = datrie.Trie.load(path)
    end = time.perf_counter()
    elapsed = end - start
    print(f'{msg_prefix} Loaded {path} in {elapsed:.4f} seconds.')
    return trie


def load_wiki_tries_parallel(
    directory: str, max_threads: int = 0
) -> List[datrie.Trie]:
    if max_threads < 1:
        max_threads = os.cpu_count()

    tries = [None] * PARTITIONS
    paths_by_first_char = get_wiki_tries(directory)
    print(
        f'Loading {len(paths_by_first_char)} tries in parallel with '
        f'{max_threads} threads...'
    )
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(load_trie, path): (char, path)
            for (char, path) in paths_by_first_char.items()
        }
        for future in as_completed(futures):
            (char, path) = futures[future]
            try:
                trie = future.result()
                assert trie is not None, f'Failed to load {path}'
                assert ord(char) not in tries, f'Duplicate trie for {char}'
                tries[ord(char)] = trie
            except Exception as e:
                print(f'Error loading {path}: {e}')
    end = time.perf_counter()
    elapsed = end - start
    print(f'Loaded {len(tries)} tries in {elapsed:.4f} seconds.')
    return tries


def get_sorted_values_from_trie(trie: datrie.Trie) -> np.ndarray:
    """
    Extracts and sorts all unique positive values from the given trie,
    returning them as a NumPy array of unsigned 64-bit integers.

    Args:

        trie (datrie.Trie): Supplies the trie from which values will be
            extracted.

    Returns:

        np.ndarray: A NumPy array of unsigned 64-bit integers containing all
            unique positive values from the trie.
    """
    # Flatten and convert to absolute values.
    values = [abs(v) for sublist in trie.values() for v in sublist]

    # Sort and remove duplicates.
    sorted_values = sorted(set(values))

    # Return as numpy array.
    return np.array(sorted_values, dtype=np.uint64)


def save_trie_values_as_raw_64bit_keys(trie: datrie.Trie, path: str) -> None:
    msg_prefix = f'[{threading.get_native_id()}]'
    start = time.perf_counter()
    print(f'{msg_prefix} Saving {path}...')
    values = get_sorted_values_from_trie(trie)
    fp = np.memmap(path, dtype='uint64', mode='w+', shape=values.shape)
    fp[:] = values[:]
    del fp
    end = time.perf_counter()
    elapsed = end - start
    print(f'{msg_prefix} Saved {path} in {elapsed:.4f} seconds.')


def save_trie_values_as_raw_64bit_keys_parallel(
    tries: List[datrie.Trie], directory: str, max_threads: int = 0
) -> None:
    if max_threads < 1:
        max_threads = os.cpu_count()

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        work = [
            (trie, f'{directory}/wiki-offsets-{i:03}.keys')
            for (i, trie) in enumerate(tries)
            if trie is not None
        ]

        futures = {
            executor.submit(
                save_trie_values_as_raw_64bit_keys, trie, path
            ): path
            for (trie, path) in work
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'Error saving {path}: {e}')
    end = time.perf_counter()
    elapsed = end - start
    print(f'Saved {len(tries)} tries in {elapsed:.4f} seconds.')


# =============================================================================
# Frequency Tries
# =============================================================================


def get_freq_tries_in_dir(directory: str) -> List[str]:
    """
    Returns a list of all frequency trie files in the given directory.

    Args:
        directory (str): Supplies the directory in which to search for
                         frequency trie files.

    Returns:
        List[str]: A list of all frequency trie files in the given directory.
    """
    return sorted(glob.glob(f'{directory}/freq_titles_*c.trie'))


def load_freq_tries_parallel(
    directory: str, max_threads: int = 0
) -> List[datrie.Trie]:
    if max_threads < 1:
        max_threads = os.cpu_count()

    paths = get_freq_tries_in_dir(directory)
    num_paths = len(paths)
    max_threads = min(max_threads, num_paths)
    tries = [None] * num_paths
    print(
        f'Loading {num_paths} tries in parallel with '
        f'{max_threads} threads...'
    )
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(load_trie, path): (i, path)
            for (i, path) in enumerate(paths)
        }
        for future in as_completed(futures):
            (i, path) = futures[future]
            try:
                trie = future.result()
                assert trie is not None, f'Failed to load {path}'
                assert i >= 0 and i <= num_paths - 1, f'Out of range index {i}'
                tries[i] = trie
            except Exception as e:
                print(f'Error loading {path}: {e}')
    end = time.perf_counter()
    elapsed = end - start
    print(f'Loaded {len(tries)} tries in {elapsed:.4f} seconds.')
    return tries


# =============================================================================
# Classes
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


# vim:set ts=8 sw=4 sts=4 tw=78 et:                                            #
