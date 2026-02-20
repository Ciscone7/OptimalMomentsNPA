"""YAML experiment config loader with CLI override support.

Provides a thin layer on top of ``argparse`` so that any script can accept
a ``--config experiment.yaml`` file.  Values in the YAML serve as defaults
that are **overridden** by explicit CLI flags.

Usage in a script::

    from config_utils import add_config_arg, parse_with_config

    def main(argv=None):
        p = argparse.ArgumentParser(...)
        add_config_arg(p)
        # ... add other arguments ...
        args, sources = parse_with_config(p, argv)
        # sources maps each dest → 'cli' | 'config' | 'default'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Infrastructure args that don't warrant a "using default" warning.
_PLUMBING = frozenset({
    "config", "name", "out_root", "resume", "force", "verbose",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a YAML config file.

    Raises on missing file, non-mapping content, or nested dicts.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config must be a YAML mapping, got {type(cfg).__name__}"
        )
    for key, value in cfg.items():
        if isinstance(value, dict):
            raise ValueError(
                f"Nested mappings not supported — "
                f"key '{key}' should be a scalar or list, got dict."
            )
    return cfg


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--config`` to an argument parser."""
    parser.add_argument(
        "--config", type=str, default=None, metavar="YAML",
        help="Path to YAML experiment config. CLI flags override config values.",
    )


def parse_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[List[str]] = None,
) -> Tuple[argparse.Namespace, Dict[str, str]]:
    """Parse arguments with optional YAML config support.

    Priority: **CLI flags  >  YAML config  >  argparse defaults**.

    After parsing, a summary is printed:
    * config keys that don't match any argument (warning)
    * parameters that fell back to their default value (warning)

    Returns
    -------
    args : argparse.Namespace
    sources : dict[str, str]
        Maps each argument dest to ``'cli'``, ``'config'``, or ``'default'``.
    """
    argv = argv if argv is not None else sys.argv[1:]

    # Detect which dests the user explicitly set on the command line.
    cli_dests = _detect_cli_dests(parser, argv)

    # Pre-parse to find --config.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    # Load YAML config (if provided).
    yaml_config: Dict[str, Any] = {}
    if pre_args.config:
        print(f"Loading config: {pre_args.config}")
        raw = load_yaml_config(pre_args.config)
        yaml_config = {k.replace("-", "_"): v for k, v in raw.items()}

    # Handle mutually-exclusive groups: if CLI provides a member,
    # remove competing members from config to avoid argparse conflicts.
    for group in parser._mutually_exclusive_groups:
        group_dests = {a.dest for a in group._group_actions}
        if cli_dests & group_dests:
            for dest in group_dests - cli_dests:
                yaml_config.pop(dest, None)

    # Inject remaining config values as parser defaults.
    known_dests = {a.dest for a in parser._actions if a.dest != "help"}
    config_applied: Set[str] = set()
    for dest, value in yaml_config.items():
        if dest in known_dests:
            parser.set_defaults(**{dest: value})
            config_applied.add(dest)
        elif dest != "config":
            print(f"  Warning: config key '{dest}' not recognized — ignoring")

    # Full parse (CLI args naturally override the defaults we just set).
    args = parser.parse_args(argv)

    # Build source map and warn about defaults.
    sources: Dict[str, str] = {}
    default_warnings: List[str] = []
    for dest in sorted(vars(args)):
        if dest == "config":
            continue
        if dest in cli_dests:
            sources[dest] = "cli"
        elif dest in config_applied:
            sources[dest] = "config"
        else:
            sources[dest] = "default"
            if dest not in _PLUMBING:
                value = getattr(args, dest)
                default_warnings.append(f"  {dest} = {value!r}")

    if default_warnings:
        print("Using default values (not in config or CLI):")
        for line in default_warnings:
            print(line)

    return args, sources


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_cli_dests(
    parser: argparse.ArgumentParser,
    argv: List[str],
) -> Set[str]:
    """Return the set of argparse dests explicitly provided on the CLI."""
    flag_to_dest: Dict[str, str] = {}
    for action in parser._actions:
        for opt in action.option_strings:
            flag_to_dest[opt] = action.dest

    provided: Set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        key = token.split("=")[0]
        if key in flag_to_dest:
            provided.add(flag_to_dest[key])
        # --no-xxx for BooleanOptionalAction
        elif key.startswith("--no-"):
            pos_key = "--" + key[5:]
            if pos_key in flag_to_dest:
                provided.add(flag_to_dest[pos_key])
    return provided
