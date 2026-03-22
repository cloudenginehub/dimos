# CHOMP: Covariant Hamiltonian Optimization for Motion Planning

## The Problem

Given a robot arm, a start configuration, and a goal configuration — find a
trajectory that gets from start to goal without hitting anything, and do it
*smoothly*.

Sampling-based planners like RRT solve this by randomly exploring configuration
space and connecting collision-free samples. They find *a* path, but it's
typically jerky, indirect, and needs post-processing.

CHOMP takes a fundamentally different approach: **treat the entire trajectory
as a single object and optimize it directly**.

---

## The Core Idea

Imagine stretching a rubber band between the start and goal configurations.
The rubber band wants to be straight (smooth), but obstacles push it away.
CHOMP formalizes this intuition.

### Two Competing Costs

Every trajectory has two costs:

**1. Smoothness Cost** — How jerky is the motion?

Measured by the integral of squared acceleration along the trajectory.
A straight line has zero acceleration cost. A zigzag path has high cost.

Mathematically, for a discrete trajectory with waypoints `q_1, q_2, ..., q_N`:

```
F_smooth = sum_i || q_{i-1} - 2*q_i + q_{i+1} ||^2
```

This is the finite-difference approximation of acceleration. We can write it
as a matrix operation: `F_smooth = trace(xi^T A xi)` where `A = K^T K` and
`K` is the tridiagonal finite-difference operator.

**2. Collision Cost** — How close is the robot to obstacles?

For each waypoint, we query how far the robot is from the nearest obstacle.
If the distance `d` is less than a safety margin `epsilon`, we penalize it:

```
c(d) = 0.5 * (epsilon - d)^2   if d < epsilon
c(d) = 0                        if d >= epsilon
```

The total collision cost sums this over all waypoints.

### The Optimization

CHOMP minimizes the weighted sum:

```
F_total = w_smooth * F_smooth + w_collision * F_collision
```

Using gradient descent. But here's the key insight that makes CHOMP special...

---

## The Covariant Gradient

Standard gradient descent would update the trajectory as:

```
xi <- xi - lr * gradient
```

But this treats each waypoint independently, producing jerky updates that
fight against the smoothness objective. CHOMP instead uses a **covariant
gradient** — it multiplies the gradient by `A^{-1}` (the inverse of the
smoothness metric):

```
xi <- xi - lr * A^{-1} @ gradient
```

Why does this matter? `A^{-1}` acts as a smoothing operator. When you multiply
a gradient by `A^{-1}`, it spreads local corrections across the trajectory in
a smooth way. Instead of one waypoint jumping to avoid an obstacle, the entire
trajectory curves smoothly around it.

This is analogous to natural gradient descent in optimization — it accounts
for the geometry of the space you're optimizing over.

```
Standard gradient:              Covariant gradient:

    *--*                            *
   /    \                          / \
  *      *   <- jerky local       *   *   <- smooth global
  |      |      correction       /     \     correction
  *      *                      *       *
   \    /                        \     /
    *--*                          *---*
```

---

## How Our Implementation Works

### Initialization

We start with the simplest possible trajectory: a straight line in joint space
from start to goal, discretized into `N` waypoints (default: 50).

```python
xi = np.linspace(q_start, q_goal, n_waypoints)  # shape (N, n_joints)
```

For the XArm7, each waypoint is a 7-dimensional vector (one value per joint).

### Pre-computation

We build the smoothness matrix `A` and its inverse `A^{-1}` once, since they
depend only on the number of waypoints (not the trajectory values):

```python
# K: tridiagonal finite-difference operator
K[i, i-1] = 1,  K[i, i] = -2,  K[i, i+1] = 1

# A: positive-definite smoothness metric
A = K^T @ K + regularization

# A_inv: for covariant gradient
A_inv = inv(A)
```

### Optimization Loop

Each iteration:

**Step 1: Compute collision cost + gradient**

For each interior waypoint (start and goal are pinned):
1. Set the robot to that configuration in the physics engine
2. Query `WorldSpec.get_min_distance()` — returns signed distance to nearest obstacle
3. If distance < epsilon, compute cost and numerical gradient

The numerical gradient uses forward differencing: perturb each joint by a small
delta, measure how the distance changes:

```python
for joint_j in range(n_joints):
    q_perturbed[j] += delta
    d_perturbed = world.get_min_distance(...)
    dd_dq[j] = (d_perturbed - d_base) / delta
```

**Step 2: Compute smoothness gradient**

Simple matrix multiply:
```python
grad_smooth = A @ xi_interior
```

**Step 3: Covariant update**

```python
total_grad = w_smooth * grad_smooth + w_collision * grad_collision
xi_interior -= learning_rate * (A_inv @ total_grad)
```

**Step 4: Project to joint limits**

Clip waypoints to stay within the robot's joint limits.

### Validation

After optimization converges (or times out), we validate the final trajectory
by checking every edge for collisions using `WorldSpec.check_edge_collision_free()`.

CHOMP is a local optimizer — it can get stuck in local minima. If the
trajectory still has collisions after optimization, we report failure. In
practice, this can be mitigated by:
- Seeding with an RRT solution instead of a straight line
- Running multiple restarts with perturbations
- Increasing the collision weight

---

## RRT vs CHOMP: When to Use Which

| Property | RRT-Connect | CHOMP |
|----------|-------------|-------|
| **Type** | Sampling-based | Optimization-based |
| **Completeness** | Probabilistically complete | Local optimizer (can get stuck) |
| **Path quality** | Random, needs smoothing | Inherently smooth |
| **Obstacle handling** | Works in cluttered spaces | Struggles with narrow passages |
| **Speed** | Variable (fast or slow) | Predictable iteration count |
| **Best for** | Complex obstacle environments | Smooth trajectories, open spaces |

The ideal pipeline often combines both: use RRT to find an initial feasible
path, then refine it with CHOMP for smoothness. Our implementation supports
this — you can seed CHOMP with an RRT solution.

---

## Parameters Guide

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_waypoints` | 20 | More = finer trajectory, slower optimization |
| `learning_rate` | 0.1 | Higher = faster convergence, risk of instability |
| `smoothness_weight` | 5.0 | Higher = smoother but may not avoid obstacles |
| `collision_weight` | 15.0 | Higher = stronger obstacle avoidance, longer paths |
| `collision_epsilon` | 0.03m | Safety margin around obstacles |
| `max_iterations` | 80 | Optimization budget |

### Tuning Tips

- If paths are collision-free but jerky: increase `smoothness_weight`
- If paths are smooth but clip obstacles: increase `collision_weight` or `collision_epsilon`
- If optimization is slow: reduce `n_waypoints` or `max_iterations`
- If stuck in local minima: try seeding with RRT output, or increase `learning_rate` briefly

---

## Code Architecture

```
chomp_planner.py
    CHOMPPlanner
    |-- plan_joint_path()          # Main entry point (PlannerSpec interface)
    |   |-- _validate_inputs()     # Same checks as RRT (world, collisions, limits)
    |   |-- _build_smoothness_matrices()  # K^T K and its inverse (once)
    |   |-- optimization loop:
    |   |   |-- _compute_collision_cost_and_gradient()  # WorldSpec distance queries
    |   |   |-- covariant update (A_inv @ gradient)
    |   |   |-- clip to joint limits
    |   |-- _validate_path()       # Final collision-free check
    |   |-- _trajectory_to_path()  # Convert numpy -> JointState list
    |
    |-- get_name() -> "CHOMP"
```

The planner implements `PlannerSpec` so it plugs directly into the existing
factory system:

```python
from dimos.manipulation.planning.factory import create_planner

planner = create_planner("chomp", collision_weight=100.0)
result = planner.plan_joint_path(world, robot_id, start, goal, timeout=10.0)
```

---

## References

1. Ratliff, N., Zucker, M., Bagnell, J.A., and Srinivasa, S. "CHOMP: Gradient
   optimization techniques for efficient motion planning." ICRA 2009.

2. Zucker, M., Ratliff, N., Dragan, A., Pivtoraiko, M., Klingensmith, M.,
   Dellin, C., Bagnell, J.A., and Srinivasa, S. "CHOMP: Covariant Hamiltonian
   optimization for motion planning." IJRR 2013.
