#!/usr/bin/env python3
import json, numpy as np, pinocchio as pin, uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path

TROSSEN_DIR = Path("/home/yilong/ros2_ws/src")  # adapt if needed
URDF_PATH   = TROSSEN_DIR / "trossen_arm_description/urdf/generated/wxai/wxai_base.urdf"
EE_FRAME    = "ee_gripper_link"

try:
    from pinocchio.shortcuts import buildModelsFromUrdf
    model, _, _ = buildModelsFromUrdf(str(URDF_PATH), package_dirs=[str(TROSSEN_DIR)])
except (ImportError, AttributeError):
    model = pin.buildModelFromUrdf(str(URDF_PATH))
data      = model.createData()
ee_id     = model.getFrameId(EE_FRAME)
q_current = pin.neutral(model).copy()

# IK parameters
TOL, ITERS, LAMBDA = 1e-4, 100, 1e-6

def solve_ik(target_x: float):
    global q_current
    pin.framesForwardKinematics(model, data, q_current)
    T_target = data.oMf[ee_id].copy()
    T_target.translation[0] = target_x            # only x moves
    q = q_current.copy()
    for _ in range(ITERS):
        pin.framesForwardKinematics(model, data, q)
        err = pin.log(T_target * data.oMf[ee_id].inverse())
        if np.linalg.norm(err) < TOL:
            break
        J  = pin.computeFrameJacobian(model, data, q, ee_id, pin.ReferenceFrame.LOCAL)
        dq = -np.linalg.lstsq(J + LAMBDA*np.eye(6), err, rcond=None)[0]
        q  = pin.integrate(model, q, dq)
    for j in model.joints[1:]:
        if j.nq == 1:
            q[j.idx_q] = np.clip(q[j.idx_q], j.lowerPositionLimit, j.upperPositionLimit)
    q_current = q
    return q.tolist()

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = json.loads(await ws.receive_text())  # expects {"x": float}
            angles = solve_ik(msg["x"])
            await ws.send_text(json.dumps({"q": angles}))
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)