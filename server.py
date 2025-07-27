#!/usr/bin/env python3
import numpy as np, pinocchio as pin, asyncio, json, uvicorn, os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pathlib import Path

TROSSEN_DIR = Path("/home/yilong/ros2_ws/src")
URDF_PATH   = Path('/home/yilong/ros2_ws/src/trossen_arm_description/urdf/generated/wxai/wxai_base.urdf')
EE_FRAME    = "ee_gripper_link"

model, *_   = pin.buildModelsFromUrdf(str(URDF_PATH), package_dirs=[str(TROSSEN_DIR)])
data        = model.createData()
ee_id       = model.getFrameId(EE_FRAME)
q_current   = pin.neutral(model).copy()

TOL, ITERS, LAMBDA = 1e-4, 100, 1e-6

def solve_ik(target_x: float) -> list[float]:
    """1-D IK: move gripper along x, keep y/z/orient fixed."""
    global q_current
    pin.framesForwardKinematics(model, data, q_current)
    # anchor target at current EE pose then overwrite x
    T_target        = data.oMf[ee_id].copy()
    T_target.translation[0] = target_x
    q_sol = q_current.copy()
    for _ in range(ITERS):
        pin.framesForwardKinematics(model, data, q_sol)
        err = pin.log(T_target * data.oMf[ee_id].inverse())
        if np.linalg.norm(err) < TOL:
            break
        J   = pin.computeFrameJacobian(model, data, q_sol, ee_id,
                                       pin.ReferenceFrame.LOCAL)
        dq  = -np.linalg.lstsq(J + LAMBDA*np.eye(6), err, rcond=None)[0]
        q_sol = pin.integrate(model, q_sol, dq)
    # clamp single-dof joints
    for j in model.joints[1:]:
        if j.nq == 1:
            q_sol[j.idx_q] = np.clip(q_sol[j.idx_q],
                                     j.lowerPositionLimit, j.upperPositionLimit)
    q_current = q_sol
    return q_sol.tolist()

app = FastAPI()

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_text()          # expects {"x": float}
            target_x = json.loads(msg)["x"]
            angles   = solve_ik(target_x)
            await ws.send_text(json.dumps({"q": angles}))
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
