"""
A very simple wiki search HTTP server that demonstrates useful techniques
afforded by PyParallel: the ability to load large reference data structures
into memory, and then query them as part of incoming request processing in
parallel.
"""

# =============================================================================
# Imports
# =============================================================================

import glob
import json
import logging
import mmap
import os
import sys
import tempfile
import threading
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import dirname
from typing import List, Tuple
import xml.etree.ElementTree as ET

import datrie
import mwcomposerfromhell as mwc
import mwparserfromhell as mwp
import numpy as np
from numpy import uint64

from .http.server import (
    HttpApp,
    RangedRequest,
    Request,
    date_time_string,
    route,
)

from .util import join_path, ElapsedTimer

# =============================================================================
# Configuration
# =============================================================================

if 'PARALLELOPEDIA_DATA_DIR' in os.environ:
    DATA_DIR = os.environ['PARALLELOPEDIA_DATA_DIR']
else:
    DATA_DIR = join_path(dirname(__file__), '../../data')

if 'PARALLELOPEDIA_WIKI_XML_NAME' in os.environ:
    WIKI_XML_NAME = os.environ['PARALLELOPEDIA_WIKI_XML_NAME']
else:
    WIKI_XML_NAME = 'enwiki-20150205-pages-articles.xml'

if 'PARALLELOPEDIA_WIKI_XML_DIR' in os.environ:
    WIKI_XML_DIR = os.environ['PARALLELOPEDIA_WIKI_XML_DIR']
    WIKI_XML_PATH = join_path(
        os.environ['PARALLELOPEDIA_WIKI_XML_DIR'], WIKI_XML_NAME
    )
else:
    WIKI_XML_DIR = DATA_DIR
    WIKI_XML_PATH = join_path(DATA_DIR, WIKI_XML_NAME)

# The directory where the title tries are stored.
if 'PARALLELOPEDIA_WIKI_TITLE_TRIES_DIR' in os.environ:
    TRIES_DIR = os.environ['PARALLELOPEDIA_WIKI_TITLE_TRIES_DIR']
else:
    TRIES_DIR = WIKI_XML_DIR

# This is a sorted numpy array of uint64s representing the byte offset values
# in the tries.  When given the byte offset of a title derived from a trie
# lookup, we can find the byte offset of where the next title starts within
# the xml file.  That allows us to isolate the required byte range from the
# xml file where the particular title is defined.  Such a byte range can be
# satisfied with a ranged HTTP request.
if 'PARALLELOPEDIA_WIKI_TITLES_OFFSETS_NPY_DIR' in os.environ:
    TITLES_OFFSETS_NPY_DIR = (
        os.environ['PARALLELOPEDIA_WIKI_TITLES_OFFSETS_NPY_DIR']
    )
else:
    TITLES_OFFSETS_NPY_DIR = WIKI_XML_DIR

TITLES_OFFSETS_NPY_PATH = join_path(
    TITLES_OFFSETS_NPY_DIR,
    'titles_offsets.npy',
)

# Number of partitions for the title tries.
PARTITIONS = 127

# =============================================================================
# Aliases
# =============================================================================
uint64_7 = uint64(7)
uint64_11 = uint64(11)


# =============================================================================
# Trie Helpers
# =============================================================================
def get_wiki_tries_in_dir(directory: str) -> List[str]:
    return sorted(glob.glob(f'{directory}/wiki-*.trie'))


def get_wiki_tries(directory: str) -> dict:
    paths = get_wiki_tries_in_dir(directory)
    basename = os.path.basename
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
    logging.debug(f'{msg_prefix} Loading {path}...')
    trie = datrie.Trie.load(path)
    end = time.perf_counter()
    elapsed = end - start
    logging.debug(f'{msg_prefix} Loaded {path} in {elapsed:.4f} seconds.')
    return trie


def load_wiki_tries_parallel(
    directory: str, max_threads: int = 0
) -> List[datrie.Trie]:

    if max_threads < 1:
        max_threads = os.cpu_count()

    paths_by_first_char = get_wiki_tries(directory)
    tries = [None] * PARTITIONS
    num_tries = len(paths_by_first_char)
    print(
        f'Loading {num_tries} tries in parallel with '
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
    print(f'Loaded {num_tries} tries in {elapsed:.4f} seconds.')
    return tries


# =============================================================================
# Globals
# =============================================================================
WIKI_XML_FILE = open(WIKI_XML_PATH, 'rb')
WIKI_XML_STAT = os.fstat(WIKI_XML_FILE.fileno())
WIKI_XML_SIZE = WIKI_XML_STAT.st_size
WIKI_XML_LAST_MODIFIED = date_time_string(WIKI_XML_STAT.st_mtime)
WIKI_XML_MMAP = mmap.mmap(
    WIKI_XML_FILE.fileno(),
    length=0,
    flags=mmap.MAP_SHARED,
    prot=mmap.PROT_READ,
    offset=0,
)
try:
    WIKI_XML_MMAP.madvise(mmap.MADV_RANDOM)
except AttributeError:
    # Ignore if madvise is not available.
    pass

# These are initialied in WikiApp.init_once().
TITLE_OFFSETS = None
TITLE_TRIES = None


# =============================================================================
# Misc Helpers
# =============================================================================

def json_serialization(request: Request = None, obj: dict = None) -> Request:
    """
    Helper method for converting a dict `obj` into a JSON response for the
    incoming `request`.
    """
    if not request:
        request = Request(transport=None, data=None)
    if not obj:
        obj = {}
    response = request.response
    response.code = 200
    response.message = 'OK'
    response.content_type = 'application/json; charset=UTF-8'
    response.body = json.dumps(obj)

    return request


def text_serialization(request=None, text=None):
    if not request:
        request = Request(transport=None, data=None)
    if not text:
        text = 'Hello, World!'
    response = request.response
    response.code = 200
    response.message = 'OK'
    response.content_type = 'text/plain; charset=UTF-8'
    response.body = text

    return request


# =============================================================================
# Offset Helpers
# =============================================================================
def get_page_offsets_for_key(search_string: str) -> List[Tuple[str, int, int]]:
    """
    Given a search string, return a list of tuples of the form
    (title, start_offset, end_offset) where start_offset and end_offset
    represent the byte offsets within the wiki XML file where the content
    for this title starts and ends, respectively.
    """
    if len(search_string) < 1:
        return None
    results = []
    titles = TITLE_TRIES[ord(search_string[0])]
    if not titles:
        return results
    items = titles.items(search_string)
    if not items:
        return results
    offsets = TITLE_OFFSETS
    for key, value in items:
        v = value[0]
        o = uint64(v if v > 0 else v * -1)
        ix = offsets.searchsorted(o, side='right')
        results.append((key, int(o - uint64_7), int(offsets[ix] - uint64_11)))
    return results


# =============================================================================
# MediaWiki Helpers
# =============================================================================
def wikitext_from_mediawiki_xml(xml_str: str) -> str:
    root = ET.fromstring(xml_str)
    # try namespaced <text> first, then non-namespaced
    text_el = root.find(".//{*}text")
    if text_el is None:
        text_el = root.find(".//text")
    return (text_el.text or "") if text_el is not None else ""

def mediawiki_to_html(wikitext: str) -> str:
    # call the pandoc binary (fast & deterministic)
    p = subprocess.run(
        ["pandoc", "-f", "mediawiki", "-t", "html5"],
        input=wikitext.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    return p.stdout.decode("utf-8")

# =============================================================================
# Classes
# =============================================================================

class WikiApp(HttpApp):

    @classmethod
    def init_once(cls):
        global TITLE_OFFSETS, TITLE_TRIES

        timer = ElapsedTimer()
        with timer:
            TITLE_OFFSETS = np.load(TITLES_OFFSETS_NPY_PATH)
        logging.info(f'Loaded title offsets in {timer.elapsed:.4f} seconds.')

        with timer:
            TITLE_TRIES = load_wiki_tries_parallel(TRIES_DIR)
        logging.info(f'Loaded title tries in {timer.elapsed:.4f} seconds.')

    @route
    def wiki_raw(self, request, name, **kwds):
        server = self.server
        # Do an exact lookup if we find a match.
        if len(name) < 1:
            return server.error(request, 400, "Name too short (< 1 char)")

        titles = TITLE_TRIES[ord(name[0])]
        if not titles or name not in titles:
            return server.error(request, 404)

        o = titles[name][0]
        o = uint64(o if o > 0 else o * -1)
        offsets = TITLE_OFFSETS
        ix = offsets.searchsorted(o, side='right')
        start = int(o - uint64_7)
        end = int(offsets[ix] - uint64_11)
        range_request = '%d-%d' % (start, end)
        request.range = RangedRequest(range_request)
        request.response.content_type = 'text/xml; charset=utf-8'
        return server.ranged_sendfile_mmap(
            request,
            WIKI_XML_MMAP,
            WIKI_XML_SIZE,
            WIKI_XML_LAST_MODIFIED,
        )

    @route
    def wiki(self, request, name, **kwds):
        server = self.server
        # Do an exact lookup if we find a match.
        if len(name) < 1:
            return server.error(request, 400, "Name too short (< 1 char)")

        titles = TITLE_TRIES[ord(name[0])]
        if not titles or name not in titles:
            return server.error(request, 404)

        o = titles[name][0]
        o = uint64(o if o > 0 else o * -1)
        offsets = TITLE_OFFSETS
        ix = offsets.searchsorted(o, side='right')
        start = int(o - uint64_7)
        end = int(offsets[ix] - uint64_11) + 1

        return self._send_xml_chunk_to_html(request, start, end)

    def _send_xml_chunk_to_html(self, request: Request, start: int, end: int):
        xml_chunk = WIKI_XML_MMAP[start:end]
        wikitext = wikitext_from_mediawiki_xml(xml_chunk.decode('utf-8'))
        html = mediawiki_to_html(wikitext)

        response = request.response
        response.code = 200
        response.message = 'OK'
        response.content_type = 'text/html; charset=UTF-8'
        response.content_length = len(html)
        response.body = html

        return self.server.send_response(request)

    @route
    def offsets(self, request, name, limit=None):
        server = self.server
        if not name:
            return server.error(request, 400, "Missing name")

        if len(name) < 3:
            return server.error(request, 400, "Name too short (< 3 chars)")

        return server.send_response(
            json_serialization(request, get_page_offsets_for_key(name))
        )

    @route
    def xml(self, request, *args, **kwds):
        server = self.server
        if not request.range:
            return server.error(request, 400, "Ranged-request required.")
        else:
            request.response.content_type = 'text/xml; charset=utf-8'
            return server.ranged_sendfile_mmap(
                request,
                WIKI_XML_MMAP,
                WIKI_XML_SIZE,
                WIKI_XML_LAST_MODIFIED,
            )

    @route
    def html(self, request, *args, **kwds):
        server = self.server
        rr = request.range
        if not rr:
            return server.error(request, 400, "Ranged-request required.")

        if not rr.set_file_size_safe(WIKI_XML_SIZE, self.server):
            return

        start = rr.first_byte
        end = rr.last_byte + 1
        return self._send_xml_chunk_to_html(request, start, end)

    @route
    def html_mwp(self, request, *args, **kwds):
        server = self.server
        rr = request.range
        if not rr:
            return server.error(request, 400, "Ranged-request required.")

        if not rr.set_file_size_safe(WIKI_XML_SIZE, self.server):
            return

        response = request.response
        response.code = 200
        response.message = 'OK'
        response.content_type = 'text/html; charset=UTF-8'

        file_content = WIKI_XML_MMAP[rr.first_byte:rr.last_byte + 1]

        code = mwp.parse(file_content)
        html = mwc.compose(code)
        response.content_length = len(html)
        response.body = html

        return server.send_response(request)

    @route
    def hello(self, request, *args, **kwds):
        j = {'args': args, 'kwds': kwds}
        return json_serialization(request, j)

    @route
    def title(self, request, name, *args, **kwds):
        server = self.server
        if len(name) < 1:
            return server.error(request, 400, "Name too short (< 1 char)")

        titles = TITLE_TRIES[ord(name[0])]
        if not titles or name not in titles:
            return server.error(request, 404)

        items = titles.items(name)
        return server.send_response(json_serialization(request, items))

    @route
    def json(self, request, *args, **kwds):
        return self.server.send_response(
            json_serialization(request, {'message': 'Hello, World!'})
        )

    @route
    def plaintext(self, request, *args, **kwds):
        return self.server.send_response(
            text_serialization(request, text='Hello, World!')
        )


# vim:set ts=8 sw=4 sts=4 tw=78 et:                                          #
