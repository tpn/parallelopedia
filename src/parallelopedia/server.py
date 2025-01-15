# =============================================================================
# Imports
# =============================================================================
import argparse
import asyncio
import logging
import os
from typing import List, Optional, Tuple

from parallelopedia.http.server import HttpApp, Request, make_routes, router
from parallelopedia.util import (
    get_class_from_string,
    get_classes_from_strings_parallel,
)


# =============================================================================
# Classes
# =============================================================================
class Gpt1(HttpApp):
    routes = make_routes()
    route = router(routes)

    async def generate_response(self, request: Request) -> None:
        with open(__file__, 'rb') as f:
            data = f.read()

        assert isinstance(data, bytes)

        transport = self.server.transport
        response = request.response

        response.code = 200
        response.message = 'OK'
        response.chunked_response = True
        response.content_type = 'text/plain'

        response.enable_tcp_nodelay()

        # This will just be the header for now.
        response_bytes = bytes(response)
        transport.write(response_bytes)

        await asyncio.sleep(0)

        text = data
        for c in text:
            response.send_chunk(c)
            await asyncio.sleep(0.2)

        # Send the termination chunk.
        response.send_chunk(None, transport)

        response.disable_tcp_nodelay()

    @route
    def foo(self, request: Request, name: str, **kwds: dict) -> None:
        logging.debug(f'foo: {name}')
        loop = asyncio.get_running_loop()
        loop.create_task(self.generate_response(request))

    @route
    def bar(self, request: Request, name: str, **kwds: dict) -> None:
        response = request.response
        logging.debug(f'bar: {name}')


# =============================================================================
# Main
# =============================================================================
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
        help='The protocol class to use for the HTTP server.',
    )
    parser.add_argument(
        '--app-classes',
        nargs='+',
        default=[
            'parallelopedia.gpt2.Gpt2App',
            'parallelopedia.wiki.WikiApp',
        ],
        help='Space-separated list of HTTP application classes.',
    )
    parser.add_argument(
        '--listen-backlog',
        type=int,
        default=100,
        help='The listen backlog for the server.',
    )
    args = parser.parse_args()
    return args


async def main_async(
    args: argparse.Namespace, protocol_class: type, *protocol_args: Tuple
) -> None:
    """
    This is the main function for the server when it is running in
    asynchronous mode.  It will create a server instance and then
    call serve_forever() on it.

    Arguments:
        args (argparse.Namespace): Supplies the command-line arguments.
        protocol_class (type): Supplies the protocol class to use.
        protocol_args (tuple): Supplies the arguments to pass to the
            protocol class constructor.

    """
    loop = asyncio.get_running_loop()

    server = await loop.create_server(
        lambda: protocol_class(*protocol_args),
        args.ip,
        args.port,
        backlog=args.listen_backlog,
        reuse_address=True,
        reuse_port=True,
    )

    async with server:
        await server.serve_forever()


def start_event_loop(
    args: argparse.Namespace, protocol_class: type, *protocol_args: Tuple
) -> None:
    """
    This function will start the asyncio event loop and run the main_async()
    function.  It is intended to be the target of a threading.Thread.

    Arguments:
        args (argparse.Namespace): Supplies the command-line arguments.
        protocol_class (type): Supplies the protocol class to use.
        protocol_args (tuple): Supplies the arguments to pass to the
            protocol class constructor.
    """
    asyncio.run(
        main_async(
            args,
            protocol_class,
            *protocol_args,
        ),
        debug=args.debug,
    )


def main_threaded_multi_accept(
    args: argparse.Namespace, protocol_class: type, *protocol_args: Tuple
) -> None:
    """
    This is the main function for the server when it is running in
    multi-threaded mode with multiple accept sockets.  Each thread
    will have its own asyncio loop issue a create_server() call for
    the given host/port and protocol.

    Arguments:
        args (argparse.Namespace): Supplies the command-line arguments.
        protocol_class (type): Supplies the protocol class to use.
        protocol_args (tuple): Supplies the arguments to pass to the
            protocol class constructor.
    """
    import threading

    threads = []
    for i in range(args.threads):
        thread = threading.Thread(
            target=start_event_loop,
            args=(args, protocol_class, *protocol_args),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def main():
    """
    Main entry point for parallelopedia.server module.
    """
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    # Use multiple threads to load the application classes.  As we construct
    # expensive objects in the bodies of the modules (e.g. LLM models, large
    # numpy arrays, datries, etc.), we want to parallelize this work.
    app_classes = get_classes_from_strings_parallel(args.app_classes)

    protocol_class = get_class_from_string(args.protocol_class)
    protocol_args = (app_classes,)

    if args.threads == 1:
        asyncio.run(
            main_async(
                args,
                protocol_class,
                *protocol_args,
            ),
            debug=args.debug,
        )
    else:
        main_threaded_multi_accept(args, protocol_class, *protocol_args)


if __name__ == '__main__':
    main()

# vim:set ts=8 sw=4 sts=4 tw=78 et:                                          #
