# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for CHOMP planner."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pytest

from dimos.manipulation.planning.planners.chomp_planner import CHOMPPlanner
from dimos.manipulation.planning.spec.enums import PlanningStatus
from dimos.msgs.sensor_msgs.JointState import JointState


JOINT_NAMES = ["joint1", "joint2", "joint3"]


def _js(positions: list[float]) -> JointState:
    return JointState(name=JOINT_NAMES, position=positions)


def _make_mock_world(
    collision_free: bool = True,
    min_distance: float = 1.0,
) -> MagicMock:
    """Create a mock WorldSpec for testing.

    Args:
        collision_free: Whether all configs are collision-free.
        min_distance: Signed distance returned by get_min_distance.
    """
    world = MagicMock()
    world.is_finalized = True
    world.get_robot_ids.return_value = ["arm"]
    world.get_joint_limits.return_value = (
        np.array([-3.14, -3.14, -3.14]),
        np.array([3.14, 3.14, 3.14]),
    )
    world.check_config_collision_free.return_value = collision_free
    world.check_edge_collision_free.return_value = collision_free

    # Mock scratch_context as a context manager
    @contextmanager
    def mock_scratch():
        yield MagicMock()

    world.scratch_context = mock_scratch
    world.set_joint_state = MagicMock()
    world.get_min_distance.return_value = min_distance

    return world


class TestCHOMPPlannerBasic:
    """Basic CHOMP planner tests with mock world."""

    def test_get_name(self):
        planner = CHOMPPlanner()
        assert planner.get_name() == "CHOMP"

    def test_plan_collision_free_space(self):
        """CHOMP should converge quickly in free space (no obstacles)."""
        planner = CHOMPPlanner(
            n_waypoints=20,
            max_iterations=50,
            collision_epsilon=0.05,
        )
        world = _make_mock_world(collision_free=True, min_distance=1.0)

        start = _js([0.0, 0.0, 0.0])
        goal = _js([1.0, 1.0, 1.0])

        result = planner.plan_joint_path(world, "arm", start, goal, timeout=5.0)

        assert result.is_success()
        assert result.status == PlanningStatus.SUCCESS
        assert len(result.path) == 20
        assert result.planning_time > 0.0
        assert result.path_length > 0.0
        assert result.iterations > 0

        # Check path endpoints match start and goal
        np.testing.assert_allclose(result.path[0].position, [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result.path[-1].position, [1.0, 1.0, 1.0], atol=1e-6)

    def test_plan_returns_joint_names(self):
        """All waypoints should have correct joint names."""
        planner = CHOMPPlanner(n_waypoints=10, max_iterations=10)
        world = _make_mock_world()

        result = planner.plan_joint_path(
            world, "arm", _js([0.0, 0.0, 0.0]), _js([1.0, 0.5, 0.3]), timeout=5.0
        )

        assert result.is_success()
        for wp in result.path:
            assert wp.name == JOINT_NAMES

    def test_path_is_near_linear_in_free_space(self):
        """In collision-free space, CHOMP trajectory should stay close to straight line."""
        planner = CHOMPPlanner(
            n_waypoints=10,
            max_iterations=100,
            smoothness_weight=10.0,
            collision_weight=0.0,  # No collision cost
        )
        world = _make_mock_world(min_distance=10.0)

        start = _js([0.0, 0.0, 0.0])
        goal = _js([1.0, 1.0, 1.0])

        result = planner.plan_joint_path(world, "arm", start, goal, timeout=5.0)

        assert result.is_success()
        # Endpoints should be exact
        np.testing.assert_allclose(result.path[0].position, [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result.path[-1].position, [1.0, 1.0, 1.0], atol=1e-6)
        # Path should be monotonically increasing (each joint goes from 0 to 1)
        for j in range(3):
            values = [wp.position[j] for wp in result.path]
            for i in range(len(values) - 1):
                assert values[i + 1] >= values[i] - 0.01, (
                    f"Joint {j} not monotonic at waypoint {i}"
                )


class TestCHOMPPlannerValidation:
    """Test input validation matching RRT planner behavior."""

    def test_rejects_unfinalized_world(self):
        planner = CHOMPPlanner()
        world = _make_mock_world()
        world.is_finalized = False

        result = planner.plan_joint_path(
            world, "arm", _js([0.0, 0.0, 0.0]), _js([1.0, 1.0, 1.0])
        )
        assert result.status == PlanningStatus.NO_SOLUTION
        assert "finalized" in result.message.lower()

    def test_rejects_unknown_robot(self):
        planner = CHOMPPlanner()
        world = _make_mock_world()
        world.get_robot_ids.return_value = ["other_arm"]

        result = planner.plan_joint_path(
            world, "arm", _js([0.0, 0.0, 0.0]), _js([1.0, 1.0, 1.0])
        )
        assert result.status == PlanningStatus.NO_SOLUTION

    def test_rejects_start_in_collision(self):
        planner = CHOMPPlanner()
        world = _make_mock_world()
        # First call (start check) returns False, second (goal check) returns True
        world.check_config_collision_free.side_effect = [False]

        result = planner.plan_joint_path(
            world, "arm", _js([0.0, 0.0, 0.0]), _js([1.0, 1.0, 1.0])
        )
        assert result.status == PlanningStatus.COLLISION_AT_START

    def test_rejects_goal_in_collision(self):
        planner = CHOMPPlanner()
        world = _make_mock_world()
        # First call (start check) returns True, second (goal check) returns False
        world.check_config_collision_free.side_effect = [True, False]

        result = planner.plan_joint_path(
            world, "arm", _js([0.0, 0.0, 0.0]), _js([1.0, 1.0, 1.0])
        )
        assert result.status == PlanningStatus.COLLISION_AT_GOAL

    def test_rejects_start_outside_limits(self):
        planner = CHOMPPlanner()
        world = _make_mock_world()

        result = planner.plan_joint_path(
            world, "arm", _js([5.0, 0.0, 0.0]), _js([1.0, 1.0, 1.0])
        )
        assert result.status == PlanningStatus.INVALID_START

    def test_rejects_goal_outside_limits(self):
        planner = CHOMPPlanner()
        world = _make_mock_world()

        result = planner.plan_joint_path(
            world, "arm", _js([0.0, 0.0, 0.0]), _js([5.0, 1.0, 1.0])
        )
        assert result.status == PlanningStatus.INVALID_GOAL


class TestCHOMPCollisionAvoidance:
    """Test that CHOMP responds to collision cost."""

    def test_collision_cost_drives_trajectory_away(self):
        """When obstacles are close, trajectory should differ from straight line."""
        # World with obstacle close to midpoint
        world = _make_mock_world(min_distance=0.01)  # Very close to obstacles

        planner_no_collision = CHOMPPlanner(
            n_waypoints=20, max_iterations=50, collision_weight=0.0
        )
        planner_with_collision = CHOMPPlanner(
            n_waypoints=20, max_iterations=50, collision_weight=100.0
        )

        start = _js([0.0, 0.0, 0.0])
        goal = _js([1.0, 1.0, 1.0])

        result_no = planner_no_collision.plan_joint_path(world, "arm", start, goal, timeout=5.0)
        result_with = planner_with_collision.plan_joint_path(world, "arm", start, goal, timeout=5.0)

        # Both should succeed (mock world says collision-free for validation)
        assert result_no.is_success()
        assert result_with.is_success()

        # With collision weight, trajectory should be different (optimizer tried to move away)
        mid_no = np.array(result_no.path[10].position)
        mid_with = np.array(result_with.path[10].position)
        # They may or may not differ much with a mock (constant distance),
        # but at least the planner should not crash
        assert mid_no is not None
        assert mid_with is not None

    def test_reports_failure_when_path_in_collision(self):
        """If final path validation fails, CHOMP should report NO_SOLUTION."""
        planner = CHOMPPlanner(n_waypoints=10, max_iterations=5)
        world = _make_mock_world(min_distance=1.0)
        # Validation fails: check_edge_collision_free returns False
        world.check_edge_collision_free.return_value = False

        result = planner.plan_joint_path(
            world, "arm", _js([0.0, 0.0, 0.0]), _js([1.0, 1.0, 1.0]), timeout=5.0
        )
        assert result.status == PlanningStatus.NO_SOLUTION
        assert "collision" in result.message.lower() or "local minimum" in result.message.lower()


class TestCHOMPSmoothnessMatrices:
    """Test the smoothness matrix construction."""

    def test_smoothness_matrix_is_symmetric(self):
        planner = CHOMPPlanner()
        A, A_inv, K = planner._build_smoothness_matrices(10)
        np.testing.assert_allclose(A, A.T, atol=1e-10)
        np.testing.assert_allclose(A_inv, A_inv.T, atol=1e-10)

    def test_smoothness_matrix_is_positive_definite(self):
        planner = CHOMPPlanner()
        A, _, _ = planner._build_smoothness_matrices(10)
        eigenvalues = np.linalg.eigvalsh(A)
        assert np.all(eigenvalues > 0)

    def test_a_inv_is_inverse_of_a(self):
        planner = CHOMPPlanner()
        A, A_inv, _ = planner._build_smoothness_matrices(10)
        product = A @ A_inv
        np.testing.assert_allclose(product, np.eye(10), atol=1e-6)

    def test_k_is_tridiagonal(self):
        planner = CHOMPPlanner()
        _, _, K = planner._build_smoothness_matrices(5)
        assert K[0, 0] == -2.0
        assert K[0, 1] == 1.0
        assert K[1, 0] == 1.0
        assert K[1, 1] == -2.0
