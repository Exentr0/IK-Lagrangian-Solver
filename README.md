# IK-Lagrangian-Solver

The IK-Lagrangian-Solver project presents an advanced method for solving the inverse kinematics (IK) problem for a 7-degree-of-freedom (7-DOF) robotic arm. By leveraging a Lagrangian-based optimization framework, the solver identifies joint configurations that not only achieve a desired end-effector position but also minimize a user-defined cost associated with rotating each joint. This approach facilitates the determination of the most cost-effective solution among potentially multiple feasible joint configurations.

---

## Overview

Inverse kinematics for redundant manipulators, such as a 7-DOF arm, can be challenging due to the infinite set of possible solutions. This project addresses the challenge by formulating the IK problem as an optimization task. The objective function penalizes deviations from initial joint positions in a weighted manner—where each joint’s rotation cost is specified by the user—thus enabling the method to select the solution that incurs the least “expense” in terms of joint movement.

Two implementations are provided:

1. **Constrained Implementation (with Joint Limits):**  
   Incorporates joint limit constraints using a barrier method. This version guarantees that the computed joint angles remain within their specified bounds, ensuring safe and realistic configurations.

2. **Unconstrained Implementation (without Joint Limits):**  
   Omits the enforcement of joint limits, which can be useful in scenarios where joint constraints are either managed separately or are not a concern.

---

The algorithm applies gradient descent to iteratively update both the joint angles and the Lagrange multipliers until the end-effector error falls below a specified tolerance.

### Joint Limits Enforcement

- **With Joint Limits:**  
  In this approach, inequality constraints (representing the lower and upper bounds of each joint) are enforced by incorporating a barrier term in the cost function. The solver adjusts the joint angles while penalizing movements that risk violating these bounds. Additionally, joint values are explicitly clipped to remain within their limits.

- **Without Joint Limits:**  
  Here, the solver ignores joint constraints, optimizing the objective function solely based on the cost of joint rotations and the equality constraint defined by the end-effector target position.
