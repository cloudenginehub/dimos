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

"""CHOMP (Covariant Hamiltonian Optimization for Motion Planning) planner.

Implements gradient-based trajectory optimization using only WorldSpec for
collision cost computation. No external dependencies beyond numpy.

Reference:
    Ratliff et al. "CHOMP: Gradient optimization techniques for efficient
    motion planning." ICRA 2009.
    Zucker et al. "CHOMP: Covariant Hamiltonian optimization for motion
    planning." IJRR 2013.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from dimos.manipulation.planning.spec.enums import PlanningStatus
from dimos.manipulation.planning.spec.models import PlanningResult, WorldRobotID
from dimos.manipulation.planning.spec.protocols import WorldSpec
from dimos.manipulation.planning.utils.path_utils import compute_path_length
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = setup_logger()


class CHOMPPlanner:
    """CHOMP trajectory optimizer implementing PlannerSpec.

    Optimizes a trajectory by minimizing a combined cost of smoothness
    (finite-difference acceleration) and collision avoidance (via WorldSpec
    signed distance queries). Uses covariant gradient descent for natural,
    smooth updates.

    This planner is backend-agnostic — it only uses WorldSpec methods and
    can work with any physics backend (Drake, MuJoCo, etc.).
    """

    def __init__(
        self,
        n_waypoints: int = 20,
        learning_rate: float = 0.1,
        smoothness_weight: float = 5.0,
        collision_weight: float = 15.0,
        max_iterations: int = 80,
        collision_epsilon: float = 0.03,
        joint_perturbation: float = 0.02,
        convergence_threshold: float = 1e-4,
        collision_step_size: float = 0.02,
    ):
        """Initialize CHOMP planner.

        Args:
            n_waypoints: Number of waypoints in the optimized trajectory (including endpoints).
            learning_rate: Step size for gradient descent.
            smoothness_weight: Weight for smoothness cost (penalizes acceleration).
                Higher values produce smoother trajectories.
            collision_weight: Weight for collision cost (penalizes proximity to obstacles).
                Higher values push trajectory further from obstacles.
            max_iterations: Maximum optimization iterations.
            collision_epsilon: Safety margin in meters. Waypoints closer than this
                to obstacles incur collision cost.
            joint_perturbation: Delta for numerical gradient computation (radians).
            convergence_threshold: Stop when gradient norm falls below this.
            collision_step_size: Step size for final path validation edge checks.
        """
        self._n_waypoints = n_waypoints
        self._learning_rate = learning_rate
        self._smoothness_weight = smoothness_weight
        self._collision_weight = collision_weight
        self._max_iterations = max_iterations
        self._collision_epsilon = collision_epsilon
        self._joint_perturbation = joint_perturbation
        self._convergence_threshold = convergence_threshold
        self._collision_step_size = collision_step_size

    def plan_joint_path(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        start: JointState,
        goal: JointState,
        timeout: float = 10.0,
    ) -> PlanningResult:
        """Plan a collision-free path using CHOMP trajectory optimization.

        Initializes a straight-line trajectory from start to goal, then
        iteratively optimizes it to minimize smoothness + collision cost.
        """
        start_time = time.time()

        error = self._validate_inputs(world, robot_id, start, goal)
        if error is not None:
            return error

        q_start = np.array(start.position, dtype=np.float64)
        q_goal = np.array(goal.position, dtype=np.float64)
        joint_names = list(start.name)
        n_joints = len(joint_names)
        lower, upper = world.get_joint_limits(robot_id)

        # 1. Initialize trajectory: linear interpolation
        n = self._n_waypoints
        xi = np.zeros((n, n_joints))
        for j in range(n_joints):
            xi[:, j] = np.linspace(q_start[j], q_goal[j], n)

        # 2. Build smoothness cost matrix for interior waypoints
        # K is the finite-difference acceleration operator: K[i] = xi[i-1] - 2*xi[i] + xi[i+1]
        n_interior = n - 2  # exclude pinned start/goal
        A, A_inv, K = self._build_smoothness_matrices(n_interior)

        # Boundary correction for smoothness gradient.
        # The acceleration at interior point i is: a_i = q[i-1] - 2*q[i] + q[i+1]
        # For the first interior point, q[i-1] = q_start (pinned).
        # For the last interior point, q[i+1] = q_goal (pinned).
        # The gradient dF/dxi = K^T @ (K @ xi + b) = A @ xi + K^T @ b
        # where b accounts for the pinned boundary conditions.
        b = np.zeros((n_interior, n_joints))
        b[0] = q_start
        b[-1] = q_goal
        boundary_grad = K.T @ b  # constant correction term

        # 3. Optimization loop
        best_xi = xi.copy()
        best_cost = float("inf")
        iteration = 0

        for iteration in range(self._max_iterations):
            if time.time() - start_time > timeout:
                logger.debug("CHOMP timeout after %d iterations", iteration)
                break

            # Interior waypoints only (indices 1 to n-2)
            xi_interior = xi[1:-1]  # shape (n_interior, n_joints)

            # a. Collision cost + gradient
            collision_cost, grad_collision = self._compute_collision_cost_and_gradient(
                world, robot_id, xi_interior, joint_names
            )

            # b. Smoothness cost + gradient (with boundary correction)
            # Full gradient: A @ xi_interior + K^T @ b
            grad_smooth = A @ xi_interior + boundary_grad  # (n_interior, n_joints)
            # Cost: 0.5 * ||K @ xi + b||^2
            accel = K @ xi_interior + b
            smoothness_cost = 0.5 * np.sum(accel * accel)

            total_cost = (
                self._smoothness_weight * smoothness_cost
                + self._collision_weight * collision_cost
            )

            if total_cost < best_cost:
                best_cost = total_cost
                best_xi = xi.copy()

            # c. Covariant gradient update: xi -= lr * A^{-1} @ total_gradient
            total_grad = (
                self._smoothness_weight * grad_smooth
                + self._collision_weight * grad_collision
            )

            grad_norm = float(np.linalg.norm(total_grad))
            if grad_norm < self._convergence_threshold:
                logger.debug("CHOMP converged at iteration %d (grad_norm=%.6f)", iteration, grad_norm)
                break

            # Covariant update: multiply by A^{-1} for smooth updates
            update = A_inv @ total_grad
            xi[1:-1] -= self._learning_rate * update

            # d. Project to joint limits
            xi[1:-1] = np.clip(xi[1:-1], lower, upper)

            # Pin start/goal (safety)
            xi[0] = q_start
            xi[-1] = q_goal

        planning_time = time.time() - start_time

        # Use best trajectory found
        xi = best_xi

        # 4. Validate final trajectory
        path = self._trajectory_to_path(xi, joint_names)

        if not self._validate_path(world, robot_id, path):
            return PlanningResult(
                status=PlanningStatus.NO_SOLUTION,
                path=[],
                planning_time=planning_time,
                iterations=iteration + 1,
                message="CHOMP converged but path has collisions (local minimum)",
            )

        return PlanningResult(
            status=PlanningStatus.SUCCESS,
            path=path,
            planning_time=planning_time,
            path_length=compute_path_length(path),
            iterations=iteration + 1,
            message=f"CHOMP optimized path (cost={best_cost:.4f})",
        )

    def get_name(self) -> str:
        """Get planner name."""
        return "CHOMP"

    def _validate_inputs(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        start: JointState,
        goal: JointState,
    ) -> PlanningResult | None:
        """Validate planning inputs, returns error result or None if valid."""
        if not world.is_finalized:
            return PlanningResult(
                status=PlanningStatus.NO_SOLUTION,
                message="World must be finalized before planning",
            )

        if robot_id not in world.get_robot_ids():
            return PlanningResult(
                status=PlanningStatus.NO_SOLUTION,
                message=f"Robot '{robot_id}' not found",
            )

        if not world.check_config_collision_free(robot_id, start):
            return PlanningResult(
                status=PlanningStatus.COLLISION_AT_START,
                message="Start configuration is in collision",
            )

        if not world.check_config_collision_free(robot_id, goal):
            return PlanningResult(
                status=PlanningStatus.COLLISION_AT_GOAL,
                message="Goal configuration is in collision",
            )

        lower, upper = world.get_joint_limits(robot_id)
        q_start = np.array(start.position, dtype=np.float64)
        q_goal = np.array(goal.position, dtype=np.float64)
        limit_eps = 1e-3

        if np.any(q_start < lower - limit_eps) or np.any(q_start > upper + limit_eps):
            return PlanningResult(
                status=PlanningStatus.INVALID_START,
                message="Start configuration is outside joint limits",
            )

        if np.any(q_goal < lower - limit_eps) or np.any(q_goal > upper + limit_eps):
            return PlanningResult(
                status=PlanningStatus.INVALID_GOAL,
                message="Goal configuration is outside joint limits",
            )

        return None

    def _build_smoothness_matrices(
        self, n: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Build the finite-difference acceleration cost matrix, its inverse, and K.

        The smoothness cost penalizes acceleration (second derivative) of the
        trajectory. K is the tridiagonal finite-difference operator for second
        derivatives, and A = K^T @ K is the positive-definite cost matrix.

        Args:
            n: Number of interior waypoints.

        Returns:
            (A, A_inv, K): Cost matrix, its inverse, and the finite-diff operator.
        """
        # K is (n, n) tridiagonal: K[i,i-1]=1, K[i,i]=-2, K[i,i+1]=1
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = -2.0
            if i > 0:
                K[i, i - 1] = 1.0
            if i < n - 1:
                K[i, i + 1] = 1.0

        A = K.T @ K

        # Regularize for numerical stability
        A += 1e-6 * np.eye(n)
        A_inv = np.linalg.inv(A)

        return A, A_inv, K

    def _compute_collision_cost_and_gradient(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        xi_interior: NDArray[np.float64],
        joint_names: list[str],
    ) -> tuple[float, NDArray[np.float64]]:
        """Compute collision cost and numerical gradient for interior waypoints.

        Uses WorldSpec.get_min_distance() with forward-difference numerical
        differentiation. Batches all queries into a single scratch context
        for performance (avoids per-query context creation overhead).

        Args:
            world: World for distance queries.
            robot_id: Robot to check.
            xi_interior: Interior waypoints, shape (n_interior, n_joints).
            joint_names: Joint names for constructing JointState.

        Returns:
            (cost, gradient): Total collision cost and gradient array
                with same shape as xi_interior.
        """
        n_pts, n_joints = xi_interior.shape
        total_cost = 0.0
        gradient = np.zeros_like(xi_interior)
        delta = self._joint_perturbation
        epsilon = self._collision_epsilon

        # Single scratch context for all distance queries in this iteration
        with world.scratch_context() as ctx:
            for i in range(n_pts):
                q = xi_interior[i]

                # Get base distance at this waypoint
                state = JointState(name=joint_names, position=q.tolist())
                world.set_joint_state(ctx, robot_id, state)
                d_base = world.get_min_distance(ctx, robot_id)

                # Collision cost: quadratic penalty when distance < epsilon
                if d_base < epsilon:
                    cost_i = 0.5 * (epsilon - d_base) ** 2
                    total_cost += cost_i

                    # Numerical gradient: forward difference for each joint
                    for j in range(n_joints):
                        q_perturbed = q.copy()
                        q_perturbed[j] += delta

                        state_p = JointState(
                            name=joint_names, position=q_perturbed.tolist()
                        )
                        world.set_joint_state(ctx, robot_id, state_p)
                        d_perturbed = world.get_min_distance(ctx, robot_id)

                        # d(cost)/d(q_j) = d(cost)/d(d) * d(d)/d(q_j)
                        # d(cost)/d(d) = -(epsilon - d) when d < epsilon
                        dd_dq = (d_perturbed - d_base) / delta
                        gradient[i, j] = -(epsilon - d_base) * dd_dq

        return total_cost, gradient

    def _trajectory_to_path(
        self,
        xi: NDArray[np.float64],
        joint_names: list[str],
    ) -> list[JointState]:
        """Convert trajectory matrix to list of JointState."""
        return [
            JointState(name=joint_names, position=xi[i].tolist())
            for i in range(xi.shape[0])
        ]

    def _validate_path(
        self,
        world: WorldSpec,
        robot_id: WorldRobotID,
        path: list[JointState],
    ) -> bool:
        """Validate that the entire path is collision-free."""
        for i in range(len(path) - 1):
            if not world.check_edge_collision_free(
                robot_id, path[i], path[i + 1], self._collision_step_size
            ):
                return False
        return True
