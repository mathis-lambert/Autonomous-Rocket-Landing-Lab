"""Run the freefall baseline scenario."""

from rocket_landing.infrastructure.cli.main import main

if __name__ == "__main__":
    """Launch the constant-action demo with zero throttle."""

    raise SystemExit(main(["demo", "--throttle", "0.0"]))
