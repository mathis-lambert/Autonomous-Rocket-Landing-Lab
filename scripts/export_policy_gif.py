"""Render a policy-controlled episode to an animated GIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pygame
from PIL import Image

from rocket_landing.infrastructure.config import DEFAULT_CONFIG_PATH, load_simulation_config
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteAssetLoader
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.display import create_display, enable_high_dpi
from rocket_landing.infrastructure.rendering.pygame.hud import HeadsUpDisplay, HudFonts
from rocket_landing.infrastructure.rendering.pygame.scene import PygameSimulationScene
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport
from rocket_landing.rl import (
    EnvConfig,
    RewardConfig,
    RocketLanderEnv,
    build_default_curriculum,
    find_stage_by_name,
    load_sac_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export one policy episode to an animated GIF")
    parser.add_argument("model_path", type=Path, help="Path to a saved SAC .zip model")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output GIF path",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON path for the terminal episode summary",
    )
    parser.add_argument("--stage", type=str, default="full_envelope", help="Curriculum stage name")
    parser.add_argument("--seed", type=int, default=0, help="Episode seed")
    parser.add_argument("--width", type=int, default=1280, help="Viewport width")
    parser.add_argument("--height", type=int, default=720, help="Viewport height")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=4,
        help="Capture every N environment steps",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=220,
        help="Maximum number of frames to keep in the GIF",
    )
    parser.add_argument(
        "--show-forces",
        action="store_true",
        help="Enable the force vector overlay",
    )
    parser.add_argument(
        "--hold-final-frames",
        type=int,
        default=30,
        help="Number of extra copies of the landing frame to append",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device passed to SB3",
    )
    return parser


def _surface_to_image(surface: pygame.Surface) -> Image.Image:
    array = pygame.surfarray.array3d(surface)
    array = np.transpose(array, (1, 0, 2))
    return Image.fromarray(array)


def _status_text(env: RocketLanderEnv, episode_finished: bool) -> str:
    session = env.session
    if session.last_result is None:
        return "READY"
    if session.last_result.landed:
        return "LANDED"
    if session.last_result.crashed:
        return "CRASH"
    if episode_finished:
        return "DONE"
    return "POLICY"


def main() -> int:
    args = build_parser().parse_args()
    if args.frame_stride <= 0 or args.max_frames <= 0 or args.hold_final_frames < 0:
        raise ValueError("frame-stride and max-frames must be positive")

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    enable_high_dpi()
    pygame.init()

    simulation_config = load_simulation_config(args.config)
    stage = find_stage_by_name(build_default_curriculum(simulation_config.params), args.stage)
    env = RocketLanderEnv(
        simulation_config.params,
        env_config=EnvConfig(),
        reward_config=RewardConfig(),
        default_initial_state=simulation_config.initial_state,
    )
    env.apply_curriculum_stage(stage)
    model = load_sac_model(args.model_path, device=args.device)

    viewport = Viewport(width=args.width, height=args.height)
    screen, viewport = create_display(viewport)
    assets = SpriteAssetLoader().load()
    scene = PygameSimulationScene(
        env.params,
        viewport,
        assets,
        SceneCamera(env.params, viewport),
        show_force_vectors=args.show_forces,
    )
    hud = HeadsUpDisplay(env.params, viewport)
    fonts = HudFonts.create()

    observation, _info = env.reset(seed=args.seed)
    initial_state = env.session.state
    frames: list[Image.Image] = []
    info: dict[str, object] = {}
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32)
        observation, _reward, terminated, truncated, info = env.step(action)

        if env.session.step_count % args.frame_stride != 0 and not (terminated or truncated):
            continue

        session = env.session
        frame_index = session.current_frame_index
        screen.fill((0, 0, 0))
        scene.draw(screen, session.history, frame_index, dt=1 / 60)
        hud.draw(
            screen,
            fonts,
            state=session.state,
            action=session.latest_action,
            elapsed_time=session.time,
            status_text=_status_text(env, terminated or truncated),
            steps_label=f"{session.step_count}/{session.max_steps}",
            paused=False,
            debug_forces=scene.show_force_vectors,
        )
        frames.append(_surface_to_image(screen))
        if len(frames) >= args.max_frames:
            break

    if not frames:
        raise RuntimeError("no frames were captured")
    if args.hold_final_frames > 0:
        frames.extend([frames[-1].copy() for _ in range(args.hold_final_frames)])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    palette_frames = [frame.convert("P", palette=Image.Palette.ADAPTIVE) for frame in frames]
    palette_frames[0].save(
        args.output,
        save_all=True,
        append_images=palette_frames[1:],
        duration=int(round(1000 * args.frame_stride * env._env_config.dt)),
        loop=0,
        optimize=False,
    )

    summary = info.get("episode")
    if isinstance(summary, dict):
        summary = {
            "initial_state": {
                "x": initial_state.x,
                "z": initial_state.z,
                "vx": initial_state.vx,
                "vz": initial_state.vz,
                "theta": initial_state.theta,
                "omega": initial_state.omega,
                "fuel": initial_state.fuel,
            },
            **summary,
        }
    if args.summary_output is not None and isinstance(summary, dict):
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    env.close()
    pygame.quit()

    if isinstance(summary, dict):
        print(json.dumps(summary, indent=2))
    else:
        print(json.dumps({"status": "no terminal summary captured"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
