"""Run a constant-action booster simulation."""

from rocket_landing import main

if __name__ == "__main__":
    """Launch the interactive live session with the manual controller selected."""

    raise SystemExit(main(["session", "--controller", "manual"]))
