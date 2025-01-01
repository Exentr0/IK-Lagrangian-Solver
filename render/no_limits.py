import numpy as np
import pybullet as p
import pybullet_data
import time


def ik_lagrangian(robot, target_pos, ee_idx, joint_costs, max_iter=1000, tol=1e-2, lr=0.05):
    # Qi - iteration i-th joint angle
    # Si - initial i-th joint angle
    # Ki - cost to move i-th joint (joint_costs)
    # G - goal position (target_pos)
    # FK(Q) - position of ee in Cartesian given Q
    # Minimize objective = Sum(Si - Qi)^2 * Ki subject to a constraint g = G - FK(Q) = 0

    # n_joints - number of joints in the robot arm
    n_joints = p.getNumJoints(robot)

    # Initialize lists for joint angles, joint limits, and initial joint angles
    angles, limits, init_angles = [], [], []

    # Collect initial joint angles, joint limits, and initial joint states
    for i in range(n_joints):
        j_info = p.getJointInfo(robot, i)
        if j_info[2] != p.JOINT_FIXED:
            j_state = p.getJointState(robot, i)
            angles.append(j_state[0])
            init_angles.append(j_state[0])
            limits.append((j_info[8], j_info[9]))

    angles = np.array(angles)
    init_angles = np.array(init_angles)

    # Initialize Lagrange multipliers for the constraint (end-effector position error)
    lambdas = np.zeros(3)

    # Iteratively optimize the joint angles using Lagrangian minimization
    for _ in range(max_iter):
        # Get the current end-effector position
        ee_state = p.getLinkState(robot, ee_idx)
        curr_pos = np.array(ee_state[4])

        # Compute error
        error = target_pos - curr_pos

        if np.linalg.norm(error) < tol:
            break

        # Compute the Jacobian, only consider linear part
        J_Linear, _ = p.calculateJacobian(robot, ee_idx, [0, 0, 0], angles.tolist(), [0.0] * len(angles), [0.0] * len(angles))
        J_Linear = np.array(J_Linear)

        # Compute the gradient of the Lagrangian with respect to joint angles (dL/dQ)
        # Lagrangian: L = Sum(Si - Qi)^2 * Ki - lambda * (G - FK(Q))
        # The gradient of L with respect to joint angles is:
        # dL/dQ = 2(Q - S) * K - lambda * J
        dL = 2 * ((angles - init_angles) * joint_costs / 100) - np.dot(lambdas, J_Linear)

        # Update the joint angles using gradient descent
        angles -= lr * dL

        # Update the Lagrange multipliers to enforce the constraint (end-effector position error)
        # The derivative of Lagrangian with respect to lambda is the constraint (FK(Q) - G)
        lambdas += lr * error

        # Clip joint angles
        for i, (low, high) in enumerate(limits):
            angles[i] = np.clip(angles[i], low, high)

        # Update the robot's joint states
        idx = 0
        for i in range(n_joints):
            j_info = p.getJointInfo(robot, i)
            if j_info[2] != p.JOINT_FIXED:
                p.resetJointState(robot, i, angles[idx])
                idx += 1

        p.stepSimulation()
        time.sleep(0.01)

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
            ik_lagrangian(robot, target_pos, ee_idx, j_costs)

            ee_pos = np.array(p.getLinkState(robot, ee_idx)[4])
            print("Final EE Position:", ee_pos)

        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    main()