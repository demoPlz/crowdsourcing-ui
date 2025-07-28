#!/usr/bin/env python3
"""
A minimal FastAPI app that

1. Serves the static folder (index.html, JS, URDF, meshes…).
2. Opens a WebSocket at /ws.
   • On connect  →  pushes an initial joint pose.
   • On message  →  prints / saves the joint values coming back.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import uvicorn, json, datetime, asyncio, pathlib

# ---------- configuration ----------
INITIAL_POSE = {
    "joint_0":  0.0,
    "joint_1":  0.75,
    "joint_2":  0.3,
    "joint_3": -0.4,
    "joint_4":  0.2,
    "joint_5":  0.0,
    "left_carriage_joint":  0.015,
    "right_carriage_joint": 0.015
}
SAVE_PATH = pathlib.Path("joint_logs.ndjson")  # one JSON per line
# -----------------------------------

app = FastAPI()
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # 1. Send initial pose immediately.
    await ws.send_json({"type": "initial_pose", "joints": INITIAL_POSE})

    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "joint_update":
                log_entry = {
                    "ts": datetime.datetime.utcnow().isoformat(),
                    "joints": msg["joints"],
                }
                SAVE_PATH.write_text(json.dumps(log_entry) + "\n", append=True)
                print("🔧  received:", log_entry)

    except WebSocketDisconnect:
        print("👋  client disconnected")


if __name__ == "__main__":
    print("\nYour app is available at: http://0.0.0.0:8000/static/\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)