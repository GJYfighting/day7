#!/usr/bin/env python3
"""One-command Day6 static, numerical, isolation, and live acceptance check."""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Callable
import zipfile


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

import cv2
import gymnasium
import numpy as np
from stable_baselines3 import SAC
import stable_baselines3
import torch
import yaml

from residual_env import (
    ACTION_SHAPE,
    OBSERVATION_SHAPE,
    ActionGuardError,
    ResidualActionController,
    ResidualGraspEnv,
)


REQUIRED_MARKERS = (
    "MODEL_LOAD",
    "BASE_EQUIVALENCE",
    "RESIDUAL_FINITE",
    "RESIDUAL_LIMIT",
    "LOW_PASS_FILTER",
    "FINAL_ACTION_LIMIT",
    "INVALID_ACTION_GUARD",
    "NO_CONFIDENCE_GATE",
    "DAY6_PATH_ISOLATION",
    "DAY5_UNCHANGED",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(root: Path) -> dict[str, Any]:
    records: list[tuple[str, str]] = []
    counts = {"files": 0, "directories": 0, "symlinks": 0, "other": 0}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        item_stat = path.lstat()
        mode = format(stat.S_IMODE(item_stat.st_mode), "o")
        if path.is_symlink():
            counts["symlinks"] += 1
            line = f"l\t{mode}\t{relative}\t{os.readlink(path)}\n"
        elif path.is_file():
            counts["files"] += 1
            line = f"f\t{mode}\t{relative}\t{sha256_file(path)}\n"
        elif path.is_dir():
            counts["directories"] += 1
            line = f"d\t{mode}\t{relative}\n"
        else:
            counts["other"] += 1
            line = f"o\t{mode}\t{relative}\n"
        records.append((relative, line))
    digest = hashlib.sha256()
    for _, line in sorted(records):
        digest.update(line.encode("utf-8"))
    return {"sha256": digest.hexdigest(), **counts}


def observation(ex: float = 0.003, ey: float = -0.002, ez: float = 0.001,
                yaw: float = 0.04) -> np.ndarray:
    return np.asarray(
        [ex, ey, ez, 0.0, 0.0, 0.0, 0.0, 0.0, math.sin(yaw), math.cos(yaw)],
        dtype=np.float32,
    )


class StubSpace:
    def __init__(self, shape: tuple[int, ...], low: np.ndarray, high: np.ndarray):
        self.shape = shape
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)


class StubModel:
    def __init__(self, value: Any, *, throws: bool = False):
        self.value = value
        self.throws = throws
        action_limit = np.asarray(
            [0.005, 0.005, 0.005, math.radians(5.0)], dtype=np.float32
        )
        self.action_space = StubSpace(ACTION_SHAPE, -action_limit, action_limit)
        self.observation_space = StubSpace(
            OBSERVATION_SHAPE,
            np.full(OBSERVATION_SHAPE, -np.finfo(np.float32).max, dtype=np.float32),
            np.full(OBSERVATION_SHAPE, np.finfo(np.float32).max, dtype=np.float32),
        )

    def predict(self, _observation: np.ndarray, deterministic: bool = True):
        if self.throws:
            raise RuntimeError("injected inference failure")
        return self.value, None


class FakeBaseEnv:
    def __init__(self):
        self.visual: dict[str, Any] = {"xyz": [0.32, 0.0, 0.025]}
        self.calls: list[np.ndarray] = []

    def step(self, action: np.ndarray):
        self.calls.append(np.asarray(action, dtype=np.float32).copy())
        return observation(), 0.0, False, False, {}

    def close(self):
        pass


class Results:
    def __init__(self):
        self.status: dict[str, str] = {}
        self.details: dict[str, Any] = {}

    def record(self, name: str, passed: bool, detail: Any = "") -> None:
        self.status[name] = "PASS" if passed else "FAIL"
        self.details[name] = detail

    def run(self, name: str, function: Callable[[], Any]) -> None:
        try:
            detail = function()
            self.record(name, True, detail)
        except Exception as exception:
            self.record(
                name,
                False,
                {
                    "error": f"{type(exception).__name__}: {exception}",
                    "traceback": traceback.format_exc(limit=8),
                },
            )


def assert_close(actual: np.ndarray, expected: np.ndarray,
                 atol: float = 1e-7) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=atol):
        raise AssertionError(f"actual={actual.tolist()} expected={expected.tolist()}")


def check_preflight(config: dict[str, Any]) -> dict[str, Any]:
    required = (
        "config.yaml",
        "session_env.sh",
        "residual_env.py",
        "day6_check.py",
        "day4_env.py",
        "visual_grasp_v5.py",
        "perception_v5.py",
        "perception_v5.yaml",
        "grasp_geometry_v5.py",
        "world.launch.py",
        "moveit.launch.py",
        "generated/day3_v5_robot.urdf",
        "generated/day3_v5_robot.sdf",
        "simulations/robot_gazebo/config/robot_config.yaml",
        "simulations/robot_gazebo/worlds/grasp_table.sdf",
        "moveit_v5/jetarm_6dof.srdf",
        "moveit_v5/kinematics.yaml",
        "moveit_v5/joint_limits.yaml",
        "moveit_v5/moveit_controllers.yaml",
        str(config["day6"]["model_path"]),
    )
    missing = [item for item in required if not (ROOT / item).is_file()]
    if missing:
        raise AssertionError(f"missing required files: {missing}")
    for source in ROOT.glob("*.py"):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for model_file in (ROOT / "generated").glob("*"):
        if model_file.suffix not in (".urdf", ".sdf"):
            continue
        text = model_file.read_text(encoding="utf-8")
        for uri in __import__("re").findall(r"file://([^<\"']+)", text):
            if Path(uri).is_absolute() and not Path(uri).is_file():
                raise AssertionError(f"missing model resource: {uri}")
    sdf_check = subprocess.run(
        ["ign", "sdf", "-k", str(ROOT / "generated/day3_v5_robot.sdf")],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if sdf_check.returncode != 0 or "Valid" not in (
        sdf_check.stdout + sdf_check.stderr
    ):
        raise AssertionError((sdf_check.stdout + sdf_check.stderr)[-2000:])
    expected_versions = {
        "numpy": "1.26.4",
        "opencv": "4.8.1",
        "gymnasium": "0.29.1",
        "stable_baselines3": "2.3.2",
    }
    actual_versions = {
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
    }
    if actual_versions != expected_versions:
        raise AssertionError(
            f"dependency versions {actual_versions}, expected {expected_versions}"
        )
    redirected = (
        "ROS_LOG_DIR",
        "PYTHONPYCACHEPREFIX",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "TMPDIR",
    )
    bad_env = {
        key: os.environ.get(key, "")
        for key in redirected
        if not os.environ.get(key)
        or not Path(os.environ[key]).resolve().is_relative_to(ROOT / "runtime")
    }
    if bad_env:
        raise AssertionError(f"runtime environment is not isolated: {bad_env}")
    return {"required_file_count": len(required), "versions": actual_versions}


def check_model(controller: ResidualActionController) -> dict[str, Any]:
    if not controller.model_loaded:
        raise AssertionError(controller.model_load_error)
    if tuple(controller.model.observation_space.shape) != OBSERVATION_SHAPE:
        raise AssertionError("observation dimension mismatch")
    if tuple(controller.model.action_space.shape) != ACTION_SHAPE:
        raise AssertionError("action dimension mismatch")
    raw = np.asarray(
        controller.model.predict(observation(), deterministic=True)[0],
        dtype=np.float32,
    )
    if raw.shape != ACTION_SHAPE or not np.isfinite(raw).all():
        raise AssertionError(f"bad raw model action: {raw}")
    return {
        "path": str(controller.model_path.relative_to(ROOT)),
        "sha256": sha256_file(controller.model_path),
        "observation_shape": list(OBSERVATION_SHAPE),
        "action_shape": list(ACTION_SHAPE),
        "sample_action": raw.tolist(),
    }


def check_base_equivalence(config_path: Path) -> dict[str, Any]:
    controller = ResidualActionController(config_path)
    controller.set_enabled(False)
    gain = 0.8
    limit = np.asarray([0.005, 0.005, 0.005, math.radians(5.0)], dtype=np.float32)
    samples = (
        observation(),
        observation(1.0, -1.0, 0.5, 2.0),
        observation(-0.003, 0.004, -0.002, -0.07),
    )
    for sample in samples:
        expected = np.clip(
            np.asarray(
                [
                    gain * float(sample[0]),
                    gain * float(sample[1]),
                    gain * float(sample[2]),
                    gain * math.atan2(float(sample[8]), float(sample[9])),
                ],
                dtype=np.float32,
            ),
            -limit,
            limit,
        )
        decision = controller.decide(sample)
        assert_close(controller.compute_base_action(sample), expected, atol=0.0)
        assert_close(decision.final_action, expected, atol=0.0)
        assert_close(decision.mapped_residual, np.zeros(4, np.float32), atol=0.0)
        assert_close(decision.filtered_residual, np.zeros(4, np.float32), atol=0.0)
    return {"samples": len(samples), "tolerance": 0.0}


def check_residual_finite(controller: ResidualActionController) -> dict[str, Any]:
    controller.set_enabled(True)
    controller.reset_episode()
    decision = controller.decide(observation())
    arrays = (
        decision.raw_residual,
        decision.mapped_residual,
        decision.filtered_residual,
        decision.final_action,
    )
    if not decision.residual_valid or not all(np.isfinite(item).all() for item in arrays):
        raise AssertionError(decision.as_dict())
    return decision.as_dict()


def check_residual_limit(config_path: Path) -> dict[str, Any]:
    controller = ResidualActionController(
        config_path, model=StubModel(np.asarray([1e6, -1e6, 1e6, -1e6]))
    )
    decision = controller.decide(observation(0, 0, 0, 0))
    expected = np.asarray(
        [0.001, -0.001, 0.001, -math.radians(1.0)], dtype=np.float32
    )
    assert_close(decision.mapped_residual, expected)
    if np.any(np.abs(decision.mapped_residual) > controller.residual_limit + 1e-8):
        raise AssertionError("mapped residual exceeded configured limit")
    return {
        "mapped": decision.mapped_residual.tolist(),
        "limit": controller.residual_limit.tolist(),
    }


def check_low_pass(config_path: Path) -> dict[str, Any]:
    model_action = np.asarray([0.005, 0.005, 0.005, math.radians(5)], np.float32)
    controller = ResidualActionController(config_path, model=StubModel(model_action))
    zero = observation(0, 0, 0, 0)
    first = controller.decide(zero).filtered_residual
    second = controller.decide(zero).filtered_residual
    expected_first = controller.alpha * controller.residual_limit
    expected_second = (
        controller.alpha * controller.residual_limit
        + (1.0 - controller.alpha) * expected_first
    )
    assert_close(first, expected_first)
    assert_close(second, expected_second)
    controller.reset_episode()
    after_reset = controller.decide(zero).filtered_residual
    assert_close(after_reset, expected_first)
    return {
        "alpha": controller.alpha,
        "first": first.tolist(),
        "second": second.tolist(),
        "after_reset": after_reset.tolist(),
    }


def check_final_limit(config_path: Path) -> dict[str, Any]:
    high = np.asarray([0.005, 0.005, 0.005, math.radians(5)], np.float32)
    controller = ResidualActionController(config_path, model=StubModel(high))
    positive = controller.decide(observation(2, 2, 2, 2)).final_action
    assert_close(positive, controller.final_high)
    controller = ResidualActionController(config_path, model=StubModel(-high))
    negative = controller.decide(observation(-2, -2, -2, -2)).final_action
    assert_close(negative, controller.final_low)
    return {
        "positive": positive.tolist(),
        "negative": negative.tolist(),
        "limit": controller.final_limit.tolist(),
    }


def check_invalid_guard(config_path: Path) -> dict[str, Any]:
    bad_values: list[tuple[str, Any, bool]] = [
        ("nan", np.asarray([np.nan, 0.0, 0.0, 0.0], np.float32), False),
        ("inf", np.asarray([np.inf, 0.0, 0.0, 0.0], np.float32), False),
        ("wrong_dimension", np.asarray([0.0, 0.0, 0.0], np.float32), False),
        ("exception", np.zeros(4, np.float32), True),
    ]
    fallbacks: dict[str, str] = {}
    for name, value, throws in bad_values:
        controller = ResidualActionController(
            config_path, model=StubModel(value, throws=throws)
        )
        sample = observation()
        decision = controller.decide(sample)
        if decision.residual_valid:
            raise AssertionError(f"{name} was accepted")
        assert_close(decision.mapped_residual, np.zeros(4, np.float32), atol=0.0)
        assert_close(decision.filtered_residual, np.zeros(4, np.float32), atol=0.0)
        assert_close(decision.final_action, decision.base_action, atol=0.0)
        if not np.isfinite(decision.final_action).all():
            raise AssertionError(f"{name} reached executor-facing action")
        fallbacks[name] = decision.fallback_reason

    high = np.asarray([0.005, 0.005, 0.005, math.radians(5)], np.float32)
    controller = ResidualActionController(config_path, model=StubModel(high))
    controller.decide(observation(0, 0, 0, 0))
    controller.model = StubModel(np.asarray([np.nan, 0, 0, 0], np.float32))
    controller.decide(observation(0, 0, 0, 0))
    assert_close(controller._filtered, np.zeros(4, np.float32), atol=0.0)

    controller = ResidualActionController(
        config_path, model=StubModel(np.asarray([np.nan, 0, 0, 0], np.float32))
    )
    fake = FakeBaseEnv()
    wrapper = ResidualGraspEnv(config_path, controller=controller, base_env=fake)
    wrapper.observation = observation()
    wrapper.control_step()
    if len(fake.calls) != 1:
        raise AssertionError("safe base fallback was not passed exactly once")
    assert_close(fake.calls[0], controller.compute_base_action(observation()), atol=0.0)

    call_count = len(fake.calls)
    wrapper.observation = np.full(OBSERVATION_SHAPE, np.nan, np.float32)
    try:
        wrapper.control_step()
        raise AssertionError("non-finite observation was accepted")
    except ActionGuardError:
        pass
    if len(fake.calls) != call_count:
        raise AssertionError("invalid observation entered executor")

    wrapper.observation = observation()
    fake.visual = {"xyz": [99.0, 0.0, 0.025]}
    try:
        wrapper.control_step()
        raise AssertionError("out-of-workspace target was accepted")
    except ActionGuardError:
        pass
    if len(fake.calls) != call_count:
        raise AssertionError("unsafe workspace target entered executor")

    for malformed in (
        np.full(OBSERVATION_SHAPE, np.inf, np.float32),
        np.zeros(9, np.float32),
    ):
        try:
            controller.decide(malformed)
            raise AssertionError("malformed observation was accepted")
        except ActionGuardError:
            pass
    return {"fallbacks": fallbacks, "executor_calls": len(fake.calls)}


def check_no_confidence_gate() -> dict[str, Any]:
    source_path = ROOT / "residual_env.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    forbidden_identifiers: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and "confidence" in node.id.lower():
            forbidden_identifiers.append(node.id)
        elif isinstance(node, ast.Attribute) and "confidence" in node.attr.lower():
            forbidden_identifiers.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "confidence" in node.value.lower():
                forbidden_identifiers.append(node.value)
    if forbidden_identifiers or "confidence" in source.lower():
        raise AssertionError(
            f"residual fusion contains confidence reference: {forbidden_identifiers}"
        )
    return {
        "file": source_path.name,
        "ast_nodes": sum(1 for _ in ast.walk(tree)),
        "confidence_references": 0,
    }


def decoded_model_contains(model_path: Path, needle: bytes) -> list[str]:
    hits: list[str] = []
    with zipfile.ZipFile(model_path) as archive:
        for name in archive.namelist():
            payload = archive.read(name)
            if needle in payload:
                hits.append(name)
            if name == "data":
                try:
                    data = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue
                stack: list[tuple[str, Any]] = [("data", data)]
                while stack:
                    location, value = stack.pop()
                    if isinstance(value, dict):
                        for key, child in value.items():
                            stack.append((f"{location}.{key}", child))
                    elif isinstance(value, list):
                        for index, child in enumerate(value):
                            stack.append((f"{location}[{index}]", child))
                    elif isinstance(value, str) and value.startswith(":serialized:"):
                        try:
                            decoded = base64.b64decode(value.split(":", 2)[2])
                        except Exception:
                            continue
                        if needle in decoded:
                            hits.append(location)
    return hits


def check_path_isolation() -> dict[str, Any]:
    previous = ROOT.parent / ("day" + "5")
    needle = str(previous.resolve()).encode("utf-8")
    symlinks = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_symlink()
    ]
    if symlinks:
        raise AssertionError(f"Day6 contains symlinks: {symlinks[:20]}")

    text_suffixes = {
        ".py", ".yaml", ".yml", ".sh", ".launch", ".urdf", ".sdf",
        ".srdf", ".xml", ".json",
    }
    path_hits: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in ("runtime", "results", "vendor"):
            continue
        if path.suffix.lower() not in text_suffixes:
            continue
        if needle in path.read_bytes():
            path_hits.append(relative.as_posix())
    model_hits = decoded_model_contains(ROOT / "models/sac_smoke.zip", needle)
    if path_hits or model_hits:
        raise AssertionError(
            f"previous-day path found: files={path_hits}, model={model_hits}"
        )

    local_modules = ("residual_env", "day4_env", "visual_grasp_v5", "grasp_geometry_v5")
    module_paths: dict[str, str] = {}
    for name in local_modules:
        module = importlib.import_module(name)
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(ROOT):
            raise AssertionError(f"{name} loaded outside Day6: {module_path}")
        module_paths[name] = str(module_path.relative_to(ROOT))
    vendor_modules = {
        "numpy": np,
        "cv2": cv2,
        "gymnasium": gymnasium,
        "stable_baselines3": stable_baselines3,
    }
    for name, module in vendor_modules.items():
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(VENDOR):
            raise AssertionError(f"{name} not loaded from Day6 vendor: {module_path}")
        module_paths[name] = str(module_path.relative_to(ROOT))
    contaminated_path = [
        entry for entry in sys.path
        if entry and str(previous.resolve()) in str(Path(entry).expanduser().resolve())
    ]
    contaminated_modules = [
        name for name, module in sys.modules.items()
        if getattr(module, "__file__", None)
        and str(previous.resolve()) in str(Path(module.__file__).resolve())
    ]
    if contaminated_path or contaminated_modules:
        raise AssertionError(
            f"runtime imported previous day: path={contaminated_path}, "
            f"modules={contaminated_modules[:20]}"
        )
    return {
        "symlinks": 0,
        "absolute_path_hits": 0,
        "model_serialization_hits": 0,
        "module_paths": module_paths,
    }


def check_day5_unchanged(config: dict[str, Any]) -> dict[str, Any]:
    previous = ROOT.parent / ("day" + "5")
    actual = tree_fingerprint(previous)
    provenance = config["day6"]["provenance"]
    expected = {
        "sha256": str(provenance["day5_tree_sha256"]),
        "files": int(provenance["day5_file_count"]),
        "directories": int(provenance["day5_directory_count"]),
        "symlinks": int(provenance["day5_symlink_count"]),
        "other": 0,
    }
    if actual != expected:
        raise AssertionError(f"actual={actual}, expected={expected}")
    return actual


def live_preconditions() -> dict[str, Any]:
    required_commands = ("ros2", "ign")
    missing_commands = [name for name in required_commands if not shutil.which(name)]
    if missing_commands:
        raise RuntimeError(f"missing commands: {missing_commands}")
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is unset")
    packages = (
        "ros_ign_gazebo",
        "ros_gz_bridge",
        "controller_manager",
        "moveit_ros_move_group",
        "robot_state_publisher",
    )
    prefixes: dict[str, str] = {}
    for package in packages:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", package],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ROS package unavailable: {package}: {result.stderr.strip()}"
            )
        prefixes[package] = result.stdout.strip()
    return {
        "display": os.environ["DISPLAY"],
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "ign_partition": os.environ.get("IGN_PARTITION"),
        "packages": prefixes,
    }


def process_tail(path: Path, lines: int = 50) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process_table = subprocess.run(
        ["ps", "-eo", "pid=,ppid="],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.splitlines()
    children: dict[int, list[int]] = {}
    for line in process_table:
        fields = line.split()
        if len(fields) == 2:
            pid, parent = map(int, fields)
            children.setdefault(parent, []).append(pid)
    descendants: list[int] = []
    pending = list(children.get(process.pid, ()))
    while pending:
        child = pending.pop()
        descendants.append(child)
        pending.extend(children.get(child, ()))
    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    for pid in reversed(descendants):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def run_live_rounds(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    preconditions = live_preconditions()
    log_root = ROOT / "runtime" / "logs" / "live_check"
    log_root.mkdir(parents=True, exist_ok=True)
    launch_specs = (
        (
            "world",
            ["ros2", "launch", str(ROOT / "world.launch.py"), "gui:=false"],
        ),
        (
            "moveit",
            ["ros2", "launch", str(ROOT / "moveit.launch.py")],
        ),
        (
            "perception",
            [
                os.environ.get("DAY6_PYTHON", "/usr/bin/python3"),
                str(ROOT / "perception_v5.py"),
                "--ros-args",
                "--params-file",
                str(ROOT / "perception_v5.yaml"),
            ],
        ),
    )
    processes: list[tuple[str, subprocess.Popen[Any], Any, Path]] = []
    runtime: ResidualGraspEnv | None = None
    rounds: list[dict[str, Any]] = []
    child_environment = os.environ.copy()
    # Importing the pip OpenCV wheel points Qt at OpenCV's private plugin tree.
    # Gazebo must use the system Qt plugins instead.
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        child_environment.pop(key, None)
    try:
        for name, command in launch_specs:
            log_path = log_root / f"{name}.log"
            stream = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=child_environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            processes.append((name, process, stream, log_path))
            time.sleep(2.0)
            if process.poll() is not None:
                stream.flush()
                raise RuntimeError(
                    f"{name} exited early with {process.returncode}: "
                    + "\n".join(process_tail(log_path))
                )

        controller = ResidualActionController(config_path)
        if not controller.model_loaded:
            raise RuntimeError(f"live model load failed: {controller.model_load_error}")
        runtime = ResidualGraspEnv(config_path, controller=controller)
        seed = int(config["day6"]["live_check"]["seed"])
        fixed_center = bool(config["day6"]["live_check"]["fixed_center"])
        for round_index, enabled in enumerate((False, True)):
            controller.set_enabled(enabled)
            obs, reset_info = runtime.reset(
                seed=seed + round_index,
                options={"fixed_center": fixed_center},
            )
            if obs.shape != OBSERVATION_SHAPE or not np.isfinite(obs).all():
                raise AssertionError("live reset returned invalid observation")
            terminated = False
            truncated = False
            steps = 0
            residual_nonzero = False
            final_info: dict[str, Any] = {}
            episode_return = 0.0
            maximum_steps = int(config["day4"]["max_steps"]) + 2
            while not (terminated or truncated):
                obs, reward, terminated, truncated, final_info = runtime.control_step()
                steps += 1
                episode_return += float(reward)
                action_info = final_info.get("day6_action")
                if not isinstance(action_info, dict):
                    raise AssertionError("live action telemetry missing")
                base = np.asarray(action_info["base_action"], np.float32)
                residual = np.asarray(action_info["filtered_residual"], np.float32)
                final = np.asarray(action_info["final_action"], np.float32)
                expected = np.clip(
                    base + residual, controller.final_low, controller.final_high
                )
                assert_close(final, expected)
                if not np.isfinite(final).all():
                    raise AssertionError("non-finite live action")
                if np.any(np.abs(residual) > controller.residual_limit + 1e-7):
                    raise AssertionError("live residual exceeded limit")
                if np.any(np.abs(final) > controller.final_limit + 1e-7):
                    raise AssertionError("live final action exceeded limit")
                if enabled:
                    if not bool(action_info["residual_valid"]):
                        raise AssertionError(
                            "live SAC residual invalid: "
                            + str(action_info["fallback_reason"])
                        )
                    residual_nonzero |= bool(np.any(np.abs(residual) > 1e-9))
                else:
                    assert_close(residual, np.zeros(4, np.float32), atol=0.0)
                    assert_close(final, base, atol=0.0)
                if steps > maximum_steps:
                    raise AssertionError("live episode exceeded configured max_steps")
            if steps < 1:
                raise AssertionError("live episode executed no control step")
            if enabled and not residual_nonzero:
                raise AssertionError("enabled live episode produced no SAC residual")
            rounds.append(
                {
                    "residual_enabled": enabled,
                    "seed": seed + round_index,
                    "steps": steps,
                    "return": episode_return,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(final_info.get("success", False)),
                    "reset_ok": bool(reset_info.get("reset_ok", False)),
                    "nonzero_residual": residual_nonzero,
                }
            )
        return {"preconditions": preconditions, "rounds": rounds}
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except BaseException:
                pass
        for _name, process, _stream, _path in reversed(processes):
            stop_process(process)
        for _name, _process, stream, _path in processes:
            stream.close()


def write_report(report: dict[str, Any]) -> None:
    result_dir = ROOT / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    target = result_dir / "day6_check.json"
    temporary = result_dir / ".day6_check.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Run deterministic offline checks only; the default also runs two live rounds.",
    )
    args = parser.parse_args()
    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results = Results()

    results.run("PREFLIGHT", lambda: check_preflight(config))
    try:
        controller = ResidualActionController(config_path)
    except Exception as exception:
        controller = None
        results.record("MODEL_LOAD", False, f"{type(exception).__name__}: {exception}")
    else:
        results.run("MODEL_LOAD", lambda: check_model(controller))
    results.run("BASE_EQUIVALENCE", lambda: check_base_equivalence(config_path))
    if controller is None:
        results.record("RESIDUAL_FINITE", False, "controller construction failed")
    else:
        results.run("RESIDUAL_FINITE", lambda: check_residual_finite(controller))
    results.run("RESIDUAL_LIMIT", lambda: check_residual_limit(config_path))
    results.run("LOW_PASS_FILTER", lambda: check_low_pass(config_path))
    results.run("FINAL_ACTION_LIMIT", lambda: check_final_limit(config_path))
    results.run("INVALID_ACTION_GUARD", lambda: check_invalid_guard(config_path))
    results.run("NO_CONFIDENCE_GATE", check_no_confidence_gate)
    results.run("DAY6_PATH_ISOLATION", check_path_isolation)
    results.run("DAY5_UNCHANGED", lambda: check_day5_unchanged(config))

    if args.skip_live:
        results.status["LIVE_ROUNDS"] = "SKIP"
        results.details["LIVE_ROUNDS"] = "disabled by --skip-live"
    elif (
        results.status.get("PREFLIGHT") == "PASS"
        and all(results.status.get(name) == "PASS" for name in REQUIRED_MARKERS)
    ):
        results.run("LIVE_ROUNDS", lambda: run_live_rounds(config_path, config))
    else:
        results.status["LIVE_ROUNDS"] = "SKIP"
        results.details["LIVE_ROUNDS"] = "static prerequisite failed"

    required_pass = (
        results.status.get("PREFLIGHT") == "PASS"
        and all(results.status.get(name) == "PASS" for name in REQUIRED_MARKERS)
    )
    live_pass = args.skip_live or results.status.get("LIVE_ROUNDS") == "PASS"
    overall = required_pass and live_pass
    report = {
        "schema_version": 1,
        "generated_at_epoch": time.time(),
        "overall": "PASS" if overall else "FAIL",
        "status": results.status,
        "details": results.details,
        "parameters": {
            "fusion": "clip(base_action + low_pass(mapped_sac_residual))",
            "final_limit_m": float(config["day4"]["action_limit_m"]),
            "final_yaw_limit_deg": float(config["day4"]["yaw_action_limit_deg"]),
            "residual_limit_m": float(config["day6"]["residual_limit_m"]),
            "residual_yaw_limit_deg": float(
                config["day6"]["residual_yaw_limit_deg"]
            ),
            "low_pass_alpha": float(config["day6"]["low_pass_alpha"]),
        },
    }
    write_report(report)
    for marker in REQUIRED_MARKERS:
        print(f"{marker}={results.status.get(marker, 'FAIL')}", flush=True)
    print(f"LIVE_ROUNDS={results.status.get('LIVE_ROUNDS', 'FAIL')}", flush=True)
    print(f"DAY6_CHECK={'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 2


if __name__ == "__main__":
    sys.exit(main())
