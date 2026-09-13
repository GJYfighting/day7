#!/usr/bin/env python3
"""MoveIt move_group using the isolated Day3-v5 robot and full-pose IK."""
from pathlib import Path

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description():
    root = Path(__file__).resolve().parent
    generated = root / "generated/day3_v5_robot.urdf"
    config = root / "moveit_v5"
    required = [generated, config / "jetarm_6dof.srdf", config / "kinematics.yaml"]
    if not all(path.is_file() for path in required):
        raise RuntimeError("required Day3-v5 MoveIt artifacts are missing")
    moveit_config = (
        MoveItConfigsBuilder("jetarm_6dof", package_name="robot_moveit_config")
        .robot_description(file_path=str(generated))
        .robot_description_semantic(file_path=str(config / "jetarm_6dof.srdf"))
        .robot_description_kinematics(file_path=str(config / "kinematics.yaml"))
        .joint_limits(file_path=str(config / "joint_limits.yaml"))
        .trajectory_execution(file_path=str(config / "moveit_controllers.yaml"))
        .to_moveit_configs()
    )
    return generate_move_group_launch(moveit_config)
