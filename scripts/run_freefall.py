"""Run the freefall baseline scenario."""

from rocket_landing.infrastructure.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main(["--throttle", "0.0"]))
