#===============================================================================
# Imports
#===============================================================================
import os
import logging
import argparse

from parallelopedia.http.server import (
    HttpServer,
    Request,
    Response,
    RangedRequest,

    router,
    make_routes,
)

from typing import (
    List,
    Tuple,
    Optional,
)


#===============================================================================
# Helpers
#===============================================================================
def get_class_from_string(class_name : str) -> type:
    """
    Obtains an instance of a class object from a string representation of the
    class name, which may include the module name, e.g. `spam.eggs.Bacon`.

    Arguments:
        class_name (str): Supplies the name of the class.

    Returns:
        type: Returns the class object.

    """
    parts = class_name.split('.')
    module_name = '.'.join(parts[:-1])
    class_name = parts[-1]
    if module_name:
        module = __import__(module_name)
        for comp in parts[1:]:
            module = getattr(module, comp)
    else:
        module = globals()
    return module

#===============================================================================
# Classes
#===============================================================================
class App:
    pass

class Gpt1(App):
    routes = make_routes()
    route = router(routes)

    @route
    def foo(self, request : Request, name : str, **kwds : dict) -> None:
        logging.debug(f'foo: {name}')

    @route
    def bar(self, request : Request, name : str, **kwds : dict) -> None:
        logging.debug(f'bar: {name}')

class Gpt2(App):
    routes = make_routes()
    route = router(routes)

    @route
    def cat(self, request : Request, name : str, **kwds : dict) -> None:
        logging.debug(f'cat: {name}')

    @route
    def dog(self, request : Request, name : str, **kwds : dict) -> None:
        logging.debug(f'dog: {name}')

#===============================================================================
# Main
#===============================================================================
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
        '--protocol-classes',
        nargs='+',
        default=['Gpt1', 'Gpt2'],
        help='Space-separated list of protocol classes.',
    )
    parser.add_argument(
        '--listen-backlog',
        type=int,
        default=100,
        help='The listen backlog for the server.',
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()

    logging.basicConfig(level=args.log_level)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    protocol_classes = [
        get_class_from_string(class_name)
            for class_name in args.protocol_classes
    ]

    print(protocol_classes)


if __name__ == '__main__':
    main()

# vim:set ts=8 sw=4 sts=4 tw=78 et:                                            #
