"""The ``mario`` command: train, evaluate, record and find-route.

``main`` is the composition root. It only parses arguments here; each command imports the
heavy modules (torch, Stable Baselines3, the emulator) inside its own handler, so ``--help`` and
``import mario.cli`` stay cheap and need none of them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from mario import __version__
from mario.cli import evaluate, find_route, record, train

Handler = Callable[[argparse.Namespace], int]

COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None], Handler]] = {
    "train": (train.DESCRIPTION, train.add_arguments, train.run),
    "evaluate": (evaluate.DESCRIPTION, evaluate.add_arguments, evaluate.run),
    "record": (record.DESCRIPTION, record.add_arguments, record.run),
    "find-route": (find_route.DESCRIPTION, find_route.add_arguments, find_route.run),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the parser with one subcommand per entry of ``COMMANDS``."""
    parser = argparse.ArgumentParser(
        prog="mario",
        description="Train, evaluate and record a PPO agent that plays Super Mario Bros.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (description, add_arguments, handler) in COMMANDS.items():
        command = subparsers.add_parser(name, help=description, description=description)
        add_arguments(command)
        command.set_defaults(handler=handler)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command named in ``argv`` (``sys.argv[1:]`` when omitted).

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        The exit code: 0 on success. Errors raise ``SystemExit`` with their message.
    """
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return args.handler(args)
