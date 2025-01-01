import numpy as np
import pybullet as p
import pybullet_data
import time

# Interior point method
def ik_lagrangian_with_limits(robot, target_pos, ee_idx, joint_costs, max_iter=1000, tol=1e-2, lr=0.05, barrier_mu=1, epsilon=1e-6):
    # Lagrangian: L = Sum((Si - Qi)^2 * Ki) + λ * g(Q) + μ * h(Q)
    # Objective to minimize: Sum((Si - Qi)^2 * Ki), where
    #   - Si: initial joint angle for joint i
    #   - Qi: current joint angle for joint i
    #   - Ki: cost associated with moving joint i (joint_costs)
    #
    # Equality constraint: g(Q) = G - FK(Q) = 0, where
    #   - G: target position in Cartesian coordinates (target_pos)
    #   - FK(Q): forward kinematics result for the end effector position given angles Q
    #
    # Inequality constraint (joint limits): h(Q)r <= 0, where
    #   - h(Q) includes terms like (Qi - lower_limit) and (upper_limit - Qi)
    #
    # Barrier method term for inequality constraints:
    #   - Barrier term: μ * (-1 / (Qi - lower_limit) - 1 / (upper_limit - Qi))
    #
    # Gradients:
    #   - Gradient of objective: dL/dQi = 2 * (Qi - Si) * Ki
    #   - Gradient of equality constraint: J_Linear.T * λ, where J_Linear is the Jacobian of FK(Q)
    #   - Gradient of barrier method: d/dQi of the barrier term

    n_joints = p.getNumJoints(robot)
    angles, limits, init_angles = [], [], []

    for i in range(n_joints):
        j_info = p.getJointInfo(robot, i)
        if j_info[2] != p.JOINT_FIXED:
            j_state = p.getJointState(robot, i)
            angles.append(j_state[0])
            init_angles.append(j_state[0])
            limits.append((j_info[8], j_info[9]))  # Lower and upper limits

    angles = np.array(angles)
    init_angles = np.array(init_angles)
    lambdas = np.zeros(3)

    for _ in range(max_iter):
        ee_state = p.getLinkState(robot, ee_idx)
        curr_pos = np.array(ee_state[4])
        error = target_pos - curr_pos

        if np.linalg.norm(error) < tol:
            break

        J_Linear, _ = p.calculateJacobian(robot, ee_idx, [0, 0, 0], angles.tolist(), [0.0] * len(angles), [0.0] * len(angles))
        J_Linear = np.array(J_Linear)

        # Gradient of the objective function
        dL = 2 * ((angles - init_angles) * joint_costs / 100) - np.dot(lambdas, J_Linear)

        # Gradient of the barrier method (inequality constraints)
        barrier_grad = np.zeros_like(angles)
        for i, (low, high) in enumerate(limits):
            # Adding small epsilon to avoid division by zero
            barrier_grad[i] -= barrier_mu / (angles[i] - low + epsilon)
            barrier_grad[i] += barrier_mu / (high - angles[i] + epsilon)

        dL += barrier_grad
        angles -= lr * dL
        lambdas += lr * error

        # Enforce joint limits explicitly
        for i, (low, high) in enumerate(limits):
            angles[i] = np.clip(angles[i], low, high)

        idx = 0
        for i in range(n_joints):
            j_info = p.getJointInfo(robot, i)
            if j_info[2] != p.JOINT_FIXED:
                p.resetJointState(robot, i, angles[idx])
                idx += 1

        p.stepSimulation()
        time.sleep(0.01)

        barrier_mu *= 0.99

    return angles

###################### UTILITY FUNCTIONS FOR DEMONSTRATION #############################################
def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("../arm_urdf/xarm7.urdf", basePosition=[0, 0, 0])
    ee_idx = 6
    return robot, ee_idx

def create_sphere():
    r = 0.1
    pos = [1, 1, 1]
    s_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
    v_shape = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=[1, 0, 0, 1])
    s_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=v_shape, basePosition=pos)
    p.changeDynamics(s_id, -1, mass=0)
    p.changeDynamics(s_id, -1, linearDamping=0, angularDamping=0)
    return s_id

def create_debug_params():
    sx = p.addUserDebugParameter("Sphere X", -1, 1, 1)
    sy = p.addUserDebugParameter("Sphere Y", -1, 1, 1)
    sz = p.addUserDebugParameter("Sphere Z", -1, 1, 1)
    btn = p.addUserDebugParameter("Press IK", 0, 1, 0)
    j_params = [p.addUserDebugParameter(f"Joint {i} Price", 0, 20, 10) for i in range(7)]
    return sx, sy, sz, btn, j_params

def update_sphere_pos(s_id, sx, sy, sz):
    new_pos = [p.readUserDebugParameter(sx), p.readUserDebugParameter(sy), p.readUserDebugParameter(sz)]
    p.resetBasePositionAndOrientation(s_id, new_pos, [0, 0, 0, 1])

def main():
    robot, ee_idx = setup_simulation()
    s_id = create_sphere()
    sx, sy, sz, btn, j_params = create_debug_params()

    while True:
        update_sphere_pos(s_id, sx, sy, sz)

        if p.readUserDebugParameter(btn) == 1:
            target_pos = np.array([p.readUserDebugParameter(sx), p.readUserDebugParameter(sy), p.readUserDebugParameter(sz)])
            j_costs = np.array([p.readUserDebugParameter(param) for param in j_params])
            ik_lagrangian_with_limits(robot, target_pos, ee_idx, j_costs)

            ee_pos = np.array(p.getLinkState(robot, ee_idx)[4])
            print("Final EE Position:", ee_pos)

        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    main()
