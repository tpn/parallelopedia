# ===============================================================================
# Imports
# ===============================================================================
import argparse
import asyncio
import html
import inspect
import json
import logging
import mimetypes
import mmap
import os
import posixpath
import socket
import string
import sys
import time
import urllib
from functools import partial
from typing import List, Optional, Type

import datrie

from parallelopedia.http import (
    DEFAULT_CONTENT_TYPE,
    DEFAULT_ERROR_CONTENT_TYPE,
    DEFAULT_ERROR_MESSAGE,
    DEFAULT_RESPONSE,
    DEFAULT_SERVER_RESPONSE,
    DIRECTORY_LISTING,
    RESPONSES,
)

from parallelopedia.util import get_class_from_string

# ===============================================================================
# Aliases
# ===============================================================================
url_unquote = urllib.parse.unquote
html_escape = html.escape
normpath = posixpath.normpath

# ===============================================================================
# Globals
# ===============================================================================
IS_WINDOWS = sys.platform == 'win32'
IS_LINUX = sys.platform.startswith('linux')

# ===============================================================================
# Glue
# ===============================================================================
if not mimetypes.inited:
    mimetypes.init()

extensions_map = mimetypes.types_map.copy()
extensions_map.update(
    {
        '': 'application/octet-stream',  # Default
        '.py': 'text/plain',
        '.c': 'text/plain',
        '.h': 'text/plain',
    }
)

allowed_url_characters = string.ascii_letters + string.digits + '&#?/%_'


# ===============================================================================
# Helpers
# ===============================================================================
def keep_alive_check(f):
    def decorator(*args):
        result = f(*args)
        (obj, transport) = args[0:2]
        if not obj.keep_alive:
            transport.close()
        return result


def _quote_html(html):
    return html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def translate_path(path, base=None):
    """Translate a /-separated PATH to the local filename syntax.

    Components that mean special things to the local file system
    (e.g. drive or directory names) are ignored.  (XXX They should
    probably be diagnosed.)

    """
    # abandon query parameters
    path = path.split('?', 1)[0]
    path = path.split('#', 1)[0]
    path = normpath(url_unquote(path))
    words = path.split('/')
    words = filter(None, words)
    if not base:
        base = os.getcwd()
    path = base
    for word in words:
        drive, word = os.path.splitdrive(word)
        head, word = os.path.split(word)
        if word in (os.curdir, os.pardir):
            continue
        path = os.path.join(path, word)
    return path


def guess_type(path):
    """Guess the type of a file.

    Argument is a PATH (a filename).

    Return value is a string of the form type/subtype,
    usable for a MIME Content-type header.

    The default implementation looks the file's extension
    up in the table self.extensions_map, using application/octet-stream
    as a default; however it would be permissible (if
    slow) to look inside the data to make a better guess.

    """

    (_, ext) = posixpath.splitext(path)
    if ext in extensions_map:
        return extensions_map[ext]
    ext = ext.lower()
    if ext in extensions_map:
        return extensions_map[ext]
    else:
        return extensions_map['']


weekdayname = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

monthname = [
    None,
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
]


def date_time_string(timestamp=None):
    """Return the current date and time formatted for a message header."""
    if timestamp is None:
        timestamp = time.time()
    year, month, day, hh, mm, ss, wd, y, z = time.gmtime(timestamp)
    return "%s, %02d %3s %4d %02d:%02d:%02d GMT" % (
        weekdayname[wd],
        day,
        monthname[month],
        year,
        hh,
        mm,
        ss,
    )


gmtime = date_time_string


class NotTrie(dict):
    def longest_prefix_value(self, path):
        p = path[1:]
        if not p:
            return
        ix = p.find('/')
        if ix == -1:
            return self.get(path)
        key = path[: ix + 1]
        return self.get(key)


def make_routes(allowed: Optional[str] = None) -> datrie.Trie:
    if not allowed:
        allowed = allowed_url_characters
    import datrie

    return datrie.Trie(allowed)


def router(routes=None):
    if routes is None:
        routes = make_routes()
    _routes = routes

    class route:
        routes = _routes
        _func = None
        _funcname = None
        _path = None
        _target = None

        def __init__(self, func_or_path):
            if inspect.isfunction(func_or_path):
                self.func = func_or_path
            else:
                self.path = func_or_path

        @property
        def path(self):
            return self._path

        @path.setter
        def path(self, path):
            if path[0] != '/':
                path = '/' + self.path

            self._path = path

        @property
        def func(self):
            return self._func

        @func.setter
        def func(self, func):
            self._func = func
            self._funcname = func.__code__.co_name
            if not self._path:
                self._path = '/' + self._funcname

            self.routes[self.path] = (self._funcname, self)

        @property
        def target(self):
            return self._target

        @target.setter
        def target(self, target):
            self._target = target

        @property
        def funcname(self):
            return self._funcname

        def __get__(self, obj, objtype=None):
            if not obj:
                return self.func
            return partial(self, obj)

        def __call__(self, *_args, **_kwds):
            if not self.func:
                func = _args[0]
                self.func = func
                return self

            obj = _args[0]
            request = _args[1]

            # This will be the full path received, minus query string and
            # fragment, e.g. '/offsets/Python'.
            path = request.path[len(self.path):]

            # In the case of '/offsets/Python', that'll leave us with
            # '/Python', and we want to lop off the slash.  In the case of,
            # say, '/stats', path will be empty.
            if path and path[0] == '/':
                if len(path) > 1:
                    path = path[1:]
                else:
                    path = ''

            # And that's it, the new path is passed to the callable as the
            # second positional parameter.  If a fragment was present, pass
            # that next.  Then pass a query string as **kwds if present.
            args = []
            if path:
                path = urllib.parse.unquote(path)
                args.append(path)
            if request.fragment:
                args.append(request.fragment)

            try:
                result = self.func(obj, request, *args, **request.query)
                return result
            except TypeError as e:
                print(f'Error: {e}')
                try:
                    # Try without query string **kwds.
                    return self.func(obj, request, *args)
                except TypeError:
                    # Try without fragment.
                    try:
                        return self.func(obj, request, path)
                    except TypeError:
                        # And finally, try without path.
                        return self.func(obj, request)

    def _decorator(*args, **kwds):
        if args:
            func = args[0]
            if inspect.isfunction(func):
                return route(func)
            path = func

        def _inner(f):
            r = route(f)
            r.path = path
            return r

    return _decorator


def json_serialization(request=None, obj=None):
    if not request:
        request = Request(transport=None, data=None)
    if not obj:
        obj = {'message': 'Hello, World!'}
    response = request.response
    response.code = 200
    response.message = 'OK'
    response.content_type = 'application/json; charset=UTF-8'
    response.body = json.dumps(obj)

    return request


def text_response(request=None, text=None):
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


def html_response(request, text):
    response = request.response
    response.code = 200
    response.message = 'OK'
    response.content_type = 'text/html; charset=UTF-8'
    response.body = text

    return request


def quote_html(html):
    return html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ===============================================================================
# Classes
# ===============================================================================
class Options(dict):
    def __init__(self, values=dict()):
        assert isinstance(values, dict)
        dict.__init__(self, **values)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None


class InvalidHeaderText(Exception):
    pass


class Headers(dict):
    def __init__(self, text):
        self._text = text
        if not text:
            return
        for line in text.split(b'\r\n'):
            ix = line.find(b':')
            if ix == -1:
                raise InvalidHeaderText()
            (key, value) = (line[:ix], line[ix + 1:])
            key = key.lower().decode()
            value = value.lstrip().decode()
            self[key] = value
            self[key.replace('-', '_')] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            return None

    def __setattr__(self, name, value):
        self[name] = value


class Response:
    __slots__ = (
        'body',
        'code',
        'etag',
        'date',
        'server',
        'version',
        'request',
        'command',
        'explain',
        'message',
        'sendfile',
        'timestamp',
        'transport',
        'content_type',
        'content_range',
        'last_modified',
        'other_headers',
        'content_length',
        'chunked_response',
        '_response',
    )

    def __init__(self, request):
        self.body = ''
        self.code = 0
        self.etag = None
        self.date = None
        self.server = DEFAULT_SERVER_RESPONSE
        self.version = None
        self.request = request
        self.command = None
        self.explain = ''
        self.message = None
        self.sendfile = False
        self.timestamp = None
        self.transport = request.transport
        self.last_modified = None
        self.content_type = DEFAULT_CONTENT_TYPE
        self.content_range = None
        self.last_modified = None
        self.other_headers = []
        self.content_length = 0
        self.chunked_response = False
        self._response = None

    def __bytes__(self):
        self.date = gmtime()

        body = self.body
        code = self.code
        date = self.date
        server = self.server
        message = self.message
        content_type = self.content_type

        user_other_headers = self.other_headers
        self.other_headers = []

        connection = ''
        if not self.request.keep_alive:
            connection = 'Connection: close'

        if connection:
            self.other_headers.append(connection)

        if self.last_modified:
            lm = 'Last-Modified: %s' % self.last_modified
            self.other_headers.append(lm)

        if self.content_range:
            self.other_headers.append(self.content_range)

        bytes_body = None
        if body:
            assert not self.chunked_response
            if isinstance(body, bytes):
                bytes_body = body
                body = None

                if not self.content_length:
                    self.content_length = len(bytes_body)  # + len(rn2)
            elif not self.content_length:
                bytes_body = body.encode('UTF-8', 'replace')
                body = None
                self.content_length = len(bytes_body)  # + len(rn2)

        if self.chunked_response:
            self.other_headers.append('Transfer-Encoding: chunked')

        if self.content_length:
            assert not self.chunked_response
            content_length = 'Content-Length: %d' % self.content_length
            self.other_headers.append(content_length)
            rn2 = '\r\n'
        elif self.chunked_response:
            rn2 = '\r\n'
        else:
            content_length = 'Content-Length: 0'
            self.other_headers.append(content_length)
            rn2 = ''

        if user_other_headers:
            self.other_headers.extend(user_other_headers)

        if self.other_headers:
            other_headers = '\r\n'.join(self.other_headers)
            rn1 = '\r\n'
        else:
            rn1 = ''
            other_headers = ''

        kwds = {
            'code': code,
            'message': message,
            'server': server,
            'date': date,
            'content_type': content_type,
            'other_headers': other_headers,
            'rn1': rn1,
            'rn2': rn2,
            'body': body if body else '',
        }
        response = (DEFAULT_RESPONSE % kwds).encode('UTF-8', 'replace')

        if bytes_body:
            response += bytes_body

        self._response = response
        return response

    def _set_sockopt(self, option, value):
        if self.transport:
            sock = self.transport.get_extra_info('socket')
            if sock is not None:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)

    def enable_tcp_nodelay(self):
        self._set_sockopt(socket.TCP_NODELAY, 1)

    def disable_tcp_nodelay(self):
        self._set_sockopt(socket.TCP_NODELAY, 0)

    def send_chunk(self, chunk_or_byte):
        if chunk_or_byte is not None:
            if isinstance(chunk_or_byte, int):
                chunk = f'1\r\n{chr(chunk_or_byte)}\r\n'
            else:
                chunk = f'{len(chunk_or_byte):X}\r\n{chunk_or_byte}\r\n'
        else:
            chunk = '0\r\n\r\n'
        chunk = chunk.encode('UTF-8', 'replace')
        self.transport.write(chunk)

    def end_chunks(self):
        self.send_chunk(None)

    def _to_dict(self):
        return {
            k: getattr(self, k)
            for k in self.__slots__
            if k not in ('transport', 'request')
        }

    def _to_json(self):
        return json.dumps(self._to_dict())


class Request:
    __slots__ = (
        'data',
        'body',
        'path',
        'range',
        'query',
        'version',
        'headers',
        'command',
        'raw_path',
        'response',
        'fragment',
        'transport',
        'timestamp',
        'keep_alive',
    )

    def __init__(self, transport, data):
        self.transport = transport
        self.data = data

        self.body = None
        self.path = None
        self.range = None
        self.query = {}
        self.version = None
        self.headers = None
        self.command = None
        self.raw_path = None
        self.fragment = None
        self.timestamp = None
        self.keep_alive = False
        self.response = Response(self)

    def _to_dict(self):
        return {
            k: getattr(self, k)
            for k in self.__slots__
            if k not in ('transport', 'response')
        }

    def _to_json(self):
        return json.dumps(self._to_dict())


class InvalidRangeRequest(BaseException):
    pass


class RangeRequestTooLarge(BaseException):
    pass


class RangedRequest:
    __slots__ = (
        'first_byte',
        'last_byte',
        'suffix_length',
        # These are filled in when set_file_size() is called.
        'offset',
        'num_bytes_to_send',
        'file_size',
        'content_range',
    )

    def __init__(self, requested_range):
        self.first_byte = None
        self.last_byte = None
        self.suffix_length = None

        self.offset = None
        self.num_bytes_to_send = None
        self.file_size = None
        self.content_range = None

        try:
            r = (
                requested_range.replace(' ', '')
                .replace('bytes', '')
                .replace('=', '')
            )

            if r.startswith('-'):
                self.suffix_length = int(r[1:])
            elif r.endswith('-'):
                self.first_byte = int(r[:-1])
            else:
                pair = r.split('-')
                self.first_byte = int(pair[0])
                self.last_byte = int(pair[1])
        except Exception:
            raise InvalidRangeRequest

        if self.first_byte is not None and self.last_byte is not None:
            if self.first_byte > self.last_byte:
                raise InvalidRangeRequest

    def set_file_size(self, file_size):
        self.file_size = file_size

        if self.suffix_length is not None:
            if self.suffix_length > self.file_size:
                raise InvalidRangeRequest

            self.last_byte = file_size - 1
            self.first_byte = file_size - self.suffix_length

        else:
            if self.first_byte > file_size - 1:
                raise InvalidRangeRequest

            if not self.last_byte or self.last_byte > file_size - 1:
                self.last_byte = file_size - 1

        self.num_bytes_to_send = (self.last_byte - self.first_byte) + 1

        # Verify the chunk requested is below 2GB.
        if self.num_bytes_to_send > 2**31:
            raise RangeRequestTooLarge

        self.content_range = 'Content-Range: %d-%d/%d' % (
            self.first_byte,
            self.last_byte,
            self.file_size,
        )

    def set_file_size_safe(
        self, file_size: int, server_instance: 'HttpServer'
    ) -> bool:
        """
        Sets the file size for the ranged request.  If an exception is raised,
        an appropriate error response will be dispatched via the error()
        method of the server instance.

        Args:

            file_size (int): Supplies the size of the file in bytes.

            server_instance (HttpServer): Supplies the server instance to use
                for error dispatching.

        Returns:

            True if the file size was set successfully, otherwise False.
            If False, `server_instance.error()` will have been called with the
            appropriate error code and message.

        """
        try:
            self.set_file_size(file_size)
            return True
        except InvalidRangeRequest:
            server_instance.error(
                server_instance,
                416,
                "Invalid range request",
            )
        except RangeRequestTooLarge:
            server_instance.error(
                server_instance,
                413,
                "Range too large (>2GB)",
            )
        return False


class HttpApp:
    def __init__(self, server: 'HttpServer'):
        self.server = server


class HttpServer(asyncio.Protocol):

    def __init__(self, app_classes: Optional[List[Type[HttpApp]]] = None):
        self.routes = make_routes()
        self.apps = []
        for app_class in app_classes:
            app = app_class(server=self)
            for _, value in app.routes.items():
                (_, func) = value
                func.target = app
            self.apps.append(app)
            self.routes.update(app.routes)

    def connection_made(self, transport):
        logging.debug("Connection made with transport: %s", transport)
        self.transport = transport

    def data_received(self, data):
        logging.debug("Data received: %s", data)
        request = Request(self.transport, data)
        self.process_new_request(request)

        if False:
            if not request.keep_alive:
                request.transport.close()

            response = request.response
            if not response or response.sendfile:
                return None
            else:
                return bytes(response)

    def connection_lost(self, exc):
        if exc:
            logging.warning(f'Connection lost: {exc}')
        self.transport = None

    def process_new_request(self, request):
        raw = request.data
        ix = raw.find(b'\r\n')
        if ix == -1:
            return self.error(request, 400, "Line too long")
        (requestline, rest) = (raw[:ix], raw[ix + 2:])
        words = requestline.split()
        num_words = len(words)
        if num_words == 3:
            (command, raw_path, version) = words
            if version[:5] != b'HTTP/':
                msg = "Bad request version (%s)" % version
                return self.error(request, 400, msg)
            try:
                base_version_number = version.split(b'/', 1)[1]
                version_number = base_version_number.split(b'.')
                # RFC 2145 section 3.1 says there can be only one "." and
                #   - major and minor numbers MUST be treated as
                #      separate integers;
                #   - HTTP/2.4 is a lower version than HTTP/2.13, which in
                #      turn is lower than HTTP/12.3;
                #   - Leading zeros MUST be ignored by recipients.
                if len(version_number) != 2:
                    raise ValueError
                version_number = int(version_number[0]), int(version_number[1])
            except (ValueError, IndexError):
                msg = "Bad request version (%s)" % version
                return self.error(request, 400, msg)
            if version_number >= (1, 1):
                request.keep_alive = True
            if version_number >= (2, 0):
                msg = "Invalid HTTP Version (%s)" % base_version_number
                return self.error(request, 505, msg)

        elif num_words == 2:
            (command, raw_path) = words
            if command != b'GET':
                msg = "Bad HTTP/0.9 request type (%s)" % command
                return self.error(request, 400, msg)

        elif not words:
            request.response = None
            request.keep_alive = False
            return
        else:
            msg = "Bad request syntax (%s)" % requestline
            return self.error(request, 400, msg)

        command = command.decode()
        funcname = 'do_%s' % command
        if not hasattr(self, funcname):
            msg = 'Unsupported method (%s)' % funcname
            return self.error(request, 501, msg)

        ix = rest.rfind(b'\r\n\r\n')
        if ix == -1:
            return self.error(request, 400, "Line too long")

        raw_headers = rest[:ix]
        try:
            headers = Headers(raw_headers)
        except Exception:
            return self.error(request, 400, "Malformed headers")

        h = request.headers = headers

        version = version.decode()
        raw_path = raw_path.decode()

        # Eh, urllib.parse.urlparse doesn't work here... I took a look at the
        # source and figure it's either because of all the global caching
        # going on, or it's using a generator somewhere.  So, let's just
        # manually unpack the url as needed.  (Update: probably should review
        # this assumption now that we've supposedly fixed generators.)
        url = raw_path
        if '#' in url:
            (url, request.fragment) = url.split('#', 1)
        if '?' in url:
            (url, qs) = url.split('?', 1)
            if '&' in qs:
                pairs = qs.split('&')
            else:
                pairs = [
                    qs,
                ]

            for pair in pairs:
                # Discard anything that isn't in key=value format.
                if '=' not in pair:
                    continue
                (key, value) = pair.split('=')
                if '%' in value:
                    value = url_unquote(value)
                request.query[key] = value

        request.path = url
        request.raw_path = raw_path
        request.version = version
        request.command = command

        # IE sends through 'Keep-Alive', not 'keep-alive' like everything
        # else.
        connection = (h.connection or '').lower()
        if connection == 'close':
            request.keep_alive = False
        elif connection == 'keep-alive' or version >= 'HTTP/1.1':
            request.keep_alive = True

        if not h.range:
            # See if there's a ?range=1234-5678 and use that (handy when you
            # want to test range handling via the browser, where typing
            # /foo?range=1234-4567 is easy).
            if 'range' in request.query:
                h.range = request.query['range']

        if h.range:
            if ',' in h.range:
                # Don't permit multiple ranges.
                return self.error(
                    request, 400, "Multiple ranges not supported"
                )

            # But for anything else, the HTTP spec says to fall through and
            # process as per normal, so we just blow away the h.range header
            # in that case.
            elif h.range.count('-') != 1:
                h.range = None
            else:
                try:
                    request.range = RangedRequest(h.range)
                except InvalidRangeRequest:
                    h.range = None

        # This routing/dispatching logic is quite possibly the most horrendous
        # thing I've ever written.  On the other hand, it gets the immediate
        # job done, so eh.
        self._pre_route(request)
        func = self._dispatch(request)
        if not func:
            return self.error(request, 400, 'Unsupported Method')

        return func(request)

    def _pre_route(self, request):
        """Fiddle with request.path if necessary here."""
        return None

    def _route(self, request):
        """Override in subclass if desired.  Return a callable."""
        if not self.routes:
            return
        try:
            value = self.routes.longest_prefix_value(request.path)
            if value:
                (_, func) = value
                target = func.target
                return lambda r: func(target, r)
        except (KeyError, AttributeError):
            pass

    def _dispatch(self, request):
        func = self._route(request)
        if not func:
            func = self._simple_overload_dispatch(request)
        return func

    def _simple_overload_dispatch(self, request):
        func = None
        path = request.path
        command = request.command
        funcname = 'do_%s' % command
        overload_suffix = path.replace('/', '_')
        if overload_suffix[-1] == '_':
            overload_suffix = overload_suffix[:-1]
        overload_funcname = ''.join((command.lower(), overload_suffix))
        try:
            func = getattr(self, overload_funcname)
        except AttributeError:
            try:
                # Take off the command bit.
                overload_funcname = overload_suffix
                func = getattr(self, overload_funcname)
            except AttributeError:
                try:
                    func = getattr(self, funcname)
                except AttributeError:
                    pass

        return func

    def get_routes(self, request):
        return json_serialization(request, self.routes.keys())

    def do_HEAD(self, request):
        return self.do_GET(request)

    def do_GET(self, request):
        path = translate_path(request.path)
        logging.debug("Translated path: %s", path)
        if os.path.isdir(path):
            logging.debug("Path is a directory: %s", path)
            if not request.path.endswith('/'):
                return self.redirect(request, request.path + '/')
            found = False
            for index in ("index.html", "index.htm"):
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    found = True
                    break
            if not found:
                logging.debug("No index file found in directory: %s", path)
                return self.list_directory(request, path)

        if not os.path.exists(path):
            msg = 'File not found: %s' % path
            logging.debug(msg)
            return self.error(request, 404, msg)

        return self.sendfile(request, path)

    def do_CONNECT(self, request):
        return self.error(request, 501, "Unsupported method: CONNECT")

    def list_directory(self, request, path):
        try:
            paths = os.listdir(path)
        except os.error:
            msg = 'No permission to list directory.'
            return self.error(request, 404, msg)

        paths.sort(key=lambda a: a.lower())

        displaypath = html_escape(url_unquote(request.path))
        charset = 'utf-8'
        title = 'Directory listing for %s' % displaypath
        items = []
        item_fmt = '<li><a href="%s">%s</a></li>'

        join = os.path.join
        isdir = os.path.isdir
        islink = os.path.islink

        for name in paths:
            fullname = join(path, name)
            displayname = linkname = name

            # Append / for directories or @ for symbolic links
            if isdir(fullname):
                displayname = name + "/"
                linkname = name + "/"

            if islink(fullname):
                # Note: a link to a directory displays with @ and links with /
                displayname = name + "@"

            item = item_fmt % (url_unquote(linkname), html_escape(displayname))
            items.append(item)

        items = '\n'.join(items)
        output = DIRECTORY_LISTING % locals()

        response = request.response
        response.code = 200
        response.message = 'OK'
        response.content_type = "text/html; charset=%s" % charset
        response.body = output
        return self.send_response(request)

    def ranged_sendfile_mmap(
        self,
        request: Request,
        memory_map: mmap.mmap,
        file_size: int,
        last_modified: str,
    ) -> None:
        """
        Sends a ranged request to the client using a memory-mapped file.

        Args:

            request (Request): Supplies the request object.  The request must
                have a `range` attribute of type `RangedRequest`.

            memory_map (mmap.mmap): Supplies a memory-mapped file object.  It
                is advisable to have memory-mapped the file with
                `MADV_RANDOM` on supported platforms.

            file_size (int): Supplies the size of the file in bytes.

            last_modified (str): Supplies the last modified date of the file.
                Use `date_time_string()` to generate this value from the
                file's `st_mtime`.

        Errors:

            If the request does not have a `range` attribute, a 500 internal
            server error will be dispatched.

            If the file range is invalid, a 416 range not satisfiable error
            will be dispatched.

            If the range request is over 2GB, a 413 request entity too large
            error will be dispatched.
        """
        assert request.command == 'GET'

        if not request.range:
            return self.error(request, 500, "No range request")

        r = request.range
        if not r.set_file_size_safe(file_size, self):
            return

        response = request.response
        response.content_length = r.num_bytes_to_send
        response.last_modified = last_modified
        response.code = 206
        response.message = 'Partial Content'
        response.content_range = r.content_range

        # On Windows, TransmitFile() allows preceeding "before" bytes to be
        # send before the actual file content.  On Linux, you can kind of
        # achieve this via TCP_CORK and sendfile(), but it's not as clean.
        # As we're using memory-mapped files, we can just send the headers
        # and the file content in one go.
        file_content = memory_map[r.first_byte:r.last_byte + 1]
        response.body = file_content
        self.send_response(request)

    def sendfile(self, request: Request, path: str) -> None:
        """
        Sends a file to the client.

        Args:

            request (Request): Supplies the request object.  A ranged request
                will be assumed if the request has a range attribute of type
                `RangedRequest`.

            path (str): Supplies the path to the file to send.
        """
        if IS_WINDOWS:
            return self._sendfile_windows(request, path)
        elif hasattr(os, 'sendfile'):
            return self._sendfile_posix(request, path)
        else:
            return self._sendfile_fallback(request, path)

    def _sendfile_windows(self, request: Request, path: str) -> None:
        # Todo: use an optimal TransmitFile() implementation.
        return self._sendfile_fallback(request, path)

    def _sendfile_posix(self, request: Request, path: str) -> None:
        # Todo: use an optimal sendfile() implementation.
        return self._sendfile_fallback(request, path)

    def _sendfile_fallback(self, request: Request, path: str) -> None:
        response = request.response
        response.content_type = guess_type(path)

        try:
            st = os.stat(path)
        except FileNotFoundError:
            msg = f'File not found: {path}' % path
            return self.error(request, 404, msg)

        file_size = st.st_size
        if file_size > 2**31 and not request.range:
            msg = "File too large (>2GB); use ranged requests."
            return self.error(request, 413, msg)

        last_modified = date_time_string(st.st_mtime)

        if request.range:
            r = request.range
            try:
                r.set_file_size(file_size)
            except InvalidRangeRequest:
                return self.error(request, 416, "Invalid range request")
            except RangeRequestTooLarge:
                return self.error(request, 413, "Range too large (>2GB)")

            response.content_length = r.num_bytes_to_send
            response.code = 206
            response.message = 'Partial Content'
            response.content_range = r.content_range
        else:
            response.content_length = file_size
            response.code = 200
            response.message = 'OK'

        # Sanity check our content length is below 2GB.
        assert response.content_length < 2**31

        response.last_modified = last_modified

        if request.command == 'HEAD':
            # HEAD requests don't get a body.
            return self.send_response(request)

        # Now open the file and either read the entire contents, or the
        # specific range requested.
        try:
            with open(path, 'rb') as f:
                if request.range:
                    f.seek(r.first_byte)
                    file_content = f.read(r.num_bytes_to_send)
                else:
                    file_content = f.read()
        except FileNotFoundError:
            msg = 'File not found: %s' % path
            return self.error(request, 404, msg)
        except IOError as e:
            msg = f'Error reading file: {e}'
            return self.error(request, 500, msg)

        response.body = file_content
        self.send_response(request)

    def error(self, request, code, message=None):
        r = RESPONSES[code]
        if not message:
            message = r[0]

        logging.error("Error %d: %s", code, message)

        response = request.response
        response.code = code
        response.content_type = DEFAULT_ERROR_CONTENT_TYPE
        response.message = message
        response.explain = r[1]

        response.body = DEFAULT_ERROR_MESSAGE % {
            'code': code,
            'message': message,
            'explain': response.explain,
        }

        return self.send_response(request)

    def redirect(self, request, path):
        response = request.response
        response.other_headers.append('Location: %s' % path)
        return self.response(request, 301)

    def send_response(self, request):
        response = request.response
        if request.command == 'HEAD':
            # HEAD requests don't get a body.
            response.body = None
        if response:
            response_bytes = bytes(response)
            logging.debug(f"Sending {len(response_bytes)} byte(s) response.")
            request.transport.write(response_bytes)
        if not request.keep_alive:
            logging.debug("Closing connection.")
            request.transport.close()
        else:
            logging.debug("Keeping connection alive.")

    def response(self, request, code, message=None):
        r = RESPONSES[code]
        if not message:
            message = r[0]

        response = request.response
        response.code = code
        response.message = message
        response.explain = r[1]

        return self.send_response(request)

    @classmethod
    def merge(cls, other):
        for path, value in other.routes.items():
            cls.routes[path] = value


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run the HTTP server.')
    parser.add_argument(
        '--ip',
        type=str,
        default='0.0.0.0',
        help='IP address to bind the server to.',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='Port number to bind the server to.',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode for asyncio.',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='CRITICAL',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level.',
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of threads to use.',
    )
    parser.add_argument(
        '--protocol-class',
        type=str,
        default='parallelopedia.http.server.HttpServer',
        help='The protocol class to use for the server.',
    )
    parser.add_argument(
        '--listen-backlog',
        type=int,
        default=100,
        help='The listen backlog for the server.',
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace, protocol):
    loop = asyncio.get_running_loop()

    server = await loop.create_server(
        protocol,
        args.ip,
        args.port,
        backlog=args.listen_backlog,
        reuse_address=True,
        reuse_port=True,
    )

    async with server:
        await server.serve_forever()


def start_event_loop(args: argparse.Namespace, protocol):
    asyncio.run(
        main_async(args, protocol),
        debug=args.debug,
    )


def main_threaded_multi_accept(args: argparse.Namespace, protocol):
    """
    This is the main function for the server when it is running in
    multi-threaded mode with multiple accept sockets.  Each thread
    will have its own asyncio loop issue a create_server() call for
    the given host/port and protocol.
    """
    import threading

    threads = []
    for i in range(args.threads):
        thread = threading.Thread(
            target=start_event_loop,
            args=(args, protocol),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_arguments()

        # Set up logging configuration
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        protocol = get_class_from_string(args.protocol_class)

    if args.threads == 1:
        asyncio.run(
            main_async(args, protocol),
            debug=args.debug,
        )
    else:
        main_threaded_multi_accept(args, protocol)


if __name__ == '__main__':
    main()

# vim:set ts=8 sw=4 sts=4 tw=78 et:
