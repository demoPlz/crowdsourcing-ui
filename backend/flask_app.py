from crowd_interface import CrowdInterface
from flask import Flask
from flask_cors import CORS
from flask import Flask, jsonify, Response
from flask import request, make_response
import traceback
from pathlib import Path
import json
import os
import mimetypes
import re


def create_flask_app(crowd_interface: CrowdInterface) -> Flask:
    """Create and configure Flask app with the crowd interface"""
    app = Flask(__name__)
    CORS(app, origins=["*"], 
         allow_headers=["Content-Type", "ngrok-skip-browser-warning", "X-Session-ID"],
         methods=["GET", "POST", "OPTIONS"])
    
    @app.route("/api/get-state")
    def get_state():
        
        state = crowd_interface.get_latest_state()
        
        # Check if this is a status response (no real state)
        if isinstance(state, dict) and state.get("status"):
            # Return status response directly without processing through _state_to_json
            response = jsonify(state)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        
        # Process as a real state
        payload = crowd_interface.state_to_json(state)
        
        # Prefer text_prompt (manual or VLM), otherwise simple fallback
        text = payload.get("text_prompt")
        if isinstance(text, str) and text.strip():
            payload["prompt"] = text.strip()
        else:
            payload["prompt"] = f"Task: {crowd_interface.task_text or 'crowdsourced_task'}. What should the arm do next?"

        # Tell the frontend what to do with demo videos
        payload["demo_video"] = crowd_interface.get_demo_video_config()

        response = jsonify(payload)
        # Prevent caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    @app.route("/api/test")
    def test():
        # Count total states across all episodes
        total_states = sum(len(states) for states in crowd_interface.pending_states_by_episode.values())
        return jsonify({"message": "Flask server is working", "states_count": total_states})

    @app.route("/api/demo-video-config", methods=["GET"])
    def demo_video_config():
        """
        Lightweight config endpoint so the new frontend can fetch once on load.
        Mirrors the 'demo_video' object we also embed in /api/get-state.
        """
        try:
            return jsonify(crowd_interface.get_demo_video_config())
        except Exception as e:
            return jsonify({"enabled": False, "error": str(e)}), 500
    
    @app.route("/api/submit-goal", methods=["POST"])
    def submit_goal():
        try:
            
            # Validate request data
            data = request.get_json(force=True, silent=True)
            if data is None:
                return jsonify({"status": "error", "message": "Invalid JSON data"}), 400
            
            # Check for required fields
            required_fields = ['state_id', 'episode_id', 'joint_positions', 'gripper']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({"status": "error", "message": f"Missing required fields: {missing_fields}"}), 400
            
            # Generate or retrieve session ID from request headers or IP
            session_id = request.headers.get('X-Session-ID', request.remote_addr or 'unknown')
            
            # Record this as a response to the correct state for this session
            # The frontend now includes state_id in the request data
            crowd_interface.record_response(data)
            return jsonify({"status": "ok"})
            
        except KeyError as e:
            print(f"❌ KeyError in submit_goal (missing data field): {e}")
            return jsonify({"status": "error", "message": f"Missing required field: {e}"}), 400
        except Exception as e:
            print(f"❌ Error in submit_goal endpoint: {e}")
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/pending-states-info")
    def pending_states_info():
        """Debug endpoint to see pending states information"""
        info = crowd_interface.get_pending_states_info()
        return jsonify(info)
    
    # --- PATCH: Endpoints for monitor modal -------------------------------------

    @app.route("/api/description-bank", methods=["GET"])
    def api_description_bank():
        """
        Return the description bank for the current task as both:
          - 'entries': [{id, text, full}]
          - 'raw_text': the unparsed text (for debugging or custom parsing)
        """
        try:
            bank = crowd_interface.get_description_bank()
            return jsonify({"ok": True, "task_name": (crowd_interface.task_name() or "default"),
                            "entries": bank["entries"], "raw_text": bank["raw_text"]})
        except Exception as e:
            print(f"❌ /api/description-bank error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500


    @app.route("/api/state-details", methods=["GET"])
    def api_state_details():
        """
        Query params: episode_id=<str|int>, state_id=<int>
        Returns:
          - maincam_data_url
          - is_critical
          - flex_text_prompt
          - flex_video_id
          - description_bank / description_bank_text
        """
        try:

            ep = request.args.get("episode_id", type=int)
            sid = request.args.get("state_id", type=int)
            if ep is None or sid is None:
                return jsonify({"ok": False, "error": "episode_id and state_id are required"}), 400

            # Defaults
            flex_text = ""
            flex_video_id = None
            is_imp = False
            obs_path = None

            with crowd_interface.state_lock:
                # Prefer pending
                p_ep = crowd_interface.pending_states_by_episode.get(ep, {})
                p_info = p_ep.get(sid)
                if p_info is not None:
                    is_imp = bool(p_info.get("critical", False))
                    obs_path = p_info.get("obs_path")
                    # Text: use new field name
                    flex_text = p_info.get("text_prompt") or ""
                    # Video id: use new field name  
                    raw_vid = p_info.get("video_prompt")
                    try:
                        flex_video_id = int(raw_vid) if raw_vid is not None else None
                    except Exception:
                        flex_video_id = None
                else:
                    # Completed metadata
                    c_ep = crowd_interface.completed_states_by_episode.get(ep, {})
                    c_meta = c_ep.get(sid)
                    if c_meta is None:
                        return jsonify({"ok": False, "error": f"state {sid} not found in episode {ep}"}), 404
                    is_imp = bool(c_meta.get("critical", False))  # Use consistent field name
                    flex_text = c_meta.get("text_prompt") or ""
                    raw_vid = c_meta.get("video_prompt")
                    try:
                        flex_video_id = int(raw_vid) if raw_vid is not None else None
                    except Exception:
                        flex_video_id = None
                    man = crowd_interface.completed_states_buffer_by_episode.get(ep, {}).get(sid)
                    if isinstance(man, dict):
                        obs_path = man.get("obs_path")

            # Load maincam image (if possible)
            maincam_url = None
            if obs_path:
                obs = crowd_interface.load_obs_from_disk(obs_path)
                img = crowd_interface.load_main_cam_from_obs(obs)
                if img is not None:
                    maincam_url = crowd_interface.encode_jpeg_base64(img)

            # Description bank
            bank = crowd_interface.get_description_bank()

            return jsonify({
                "ok": True,
                "episode_id": ep,
                "state_id": sid,
                "critical": is_imp,
                "text_prompt": flex_text,
                "video_prompt": flex_video_id,
                "maincam_data_url": maincam_url,
                "description_bank": bank["entries"],
                "description_bank_text": bank["raw_text"]
            })
        except Exception as e:
            print(f"❌ /api/state-details error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/update-flex-selection", methods=["POST"])
    def api_update_flex_selection():
        """
        Body JSON:
        {
          "episode_id": <str or int>,
          "state_id": <int>,
          "video_prompt": <int>
          "text_prompt": <str>
        }
        """
        try:
            data = request.get_json(force=True, silent=True) or {}
            ep_raw = data.get("episode_id")
            if ep_raw is None:
                return jsonify({"ok": False, "error": "episode_id is required"}), 400
            ep = int(ep_raw)

            sid = data.get("state_id")
            if sid is None:
                return jsonify({"ok": False, "error": "state_id is required"}), 400
            sid = int(sid)

            vid = data.get("video_prompt")
            if vid is None:
                return jsonify({"ok": False, "error": "video_prompt is required"}), 400
            vid = int(vid)

            txt = (data.get("text_prompt") or "").strip()

            updated = False
            with crowd_interface.state_lock:
                # pending?
                p_ep = crowd_interface.pending_states_by_episode.get(ep, {})
                p_info = p_ep.get(sid)
                if p_info is not None:
                    crowd_interface.set_prompt_ready(p_info, ep, sid, txt if txt else None, vid)
                    updated = True
                else:
                    # completed metadata path  
                    c_ep = crowd_interface.completed_states_by_episode.get(ep, {})
                    c_info = c_ep.get(sid)
                    if c_info is not None:
                        # metadata mirrors - use new field names
                        if txt:
                            c_info["text_prompt"] = txt
                        c_info["video_prompt"] = vid
                        c_info["prompt_ready"] = True
                        updated = True

            if not updated:
                return jsonify({"ok": False, "error": f"state {sid} not found in episode {ep}"}), 404

            return jsonify({"ok": True, "episode_id": ep, "state_id": sid,
                            "video_prompt": vid, "text_prompt": txt or None})
        except Exception as e:
            print(f"❌ /api/update-flex-selection error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500
    
    @app.route("/api/monitor/latest-state", methods=["GET"])
    def monitor_latest_state():
        """
        Read-only monitoring endpoint for episode-based state monitoring.
        Avoid building a combined dict of all pending states on every call.
        """
        try:
            with crowd_interface.state_lock:
                current_episode = crowd_interface.current_serving_episode

                total_pending = 0
                newest_state_id = None
                newest_state_data = None
                newest_episode_id = None

                for ep_id, ep_states in crowd_interface.pending_states_by_episode.items():
                    n = len(ep_states)
                    total_pending += n
                    if n == 0:
                        continue
                    # Max by key without materializing a merged dict
                    ep_max_id = max(ep_states.keys())
                    if newest_state_id is None or ep_max_id > newest_state_id:
                        newest_state_id = ep_max_id
                        newest_state_data = ep_states[ep_max_id]
                        newest_episode_id = ep_id

                if total_pending == 0 or newest_state_data is None:
                    return jsonify({
                        "status": "no_pending_states",
                        "message": "No pending states.",
                        "views": crowd_interface.snapshot_latest_views(),  # still show previews
                        "total_pending_states": 0,
                        "current_serving_episode": current_episode,
                        "is_resetting": crowd_interface.is_in_reset(),
                        "reset_countdown": crowd_interface.get_reset_countdown()
                    })

            # Build response outside the lock
            # newest_state_data IS the state info directly (flattened structure)
            monitoring_data = {
                "status": "success",
                "state_id": newest_state_id,
                "episode_id": newest_episode_id,
                "current_serving_episode": current_episode,
                "responses_received": newest_state_data["responses_received"],
                "responses_required": (
                    crowd_interface.required_responses_per_critical_state
                    if newest_state_data.get("critical", False)
                    else crowd_interface.required_responses_per_state
                ),
                "critical": newest_state_data.get("critical", False),
                "views": crowd_interface.snapshot_latest_views(),  # lightweight snapshot (pre-encoded)
                "joint_positions": newest_state_data.get("joint_positions", {}),
                "gripper": newest_state_data.get("gripper", 0),
                "is_resetting": crowd_interface.is_in_reset(),
                "reset_countdown": crowd_interface.get_reset_countdown(),
                "total_pending_states": total_pending
            }

            response = jsonify(monitoring_data)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response

        except Exception as e:
            return jsonify({
                "status": "error",
                "message": f"Monitoring error: {str(e)}"
            }), 500
    
    @app.route("/api/control/next-episode", methods=["POST"])
    def next_episode():
        """Trigger next episode (equivalent to 'q' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Exiting current loop...")
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Next episode triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/control/rerecord", methods=["POST"])
    def rerecord_episode():
        """Trigger re-record episode (equivalent to 'r' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Exiting loop and re-record the last episode...")
                crowd_interface.events["rerecord_episode"] = True
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Re-record episode triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/control/stop", methods=["POST"])
    def stop_recording():
        """Trigger stop recording (equivalent to 'x' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Stopping data recording...")
                crowd_interface.events["stop_recording"] = True
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Stop recording triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/control/start-episode", methods=["POST"])
    def start_episode():
        """Skip remaining reset time and start the next episode immediately"""
        try:
            if crowd_interface.is_in_reset():
                crowd_interface.stop_reset()
                return jsonify({"status": "success", "message": "Reset skipped, starting episode"})
            else:
                return jsonify({"status": "error", "message": "Not currently in reset state"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route("/api/save-calibration", methods=["POST"])
    def save_calibration():
        """
        Save manual calibration to ../calib/manual_calibration_{camera}.json
        Also updates the in-memory camera models/poses so the user immediately sees results.
        Expected JSON:
        {
          "camera": "front",
          "intrinsics": {"width": W, "height": H, "Knew": [[fx,0,cx],[0,fy,cy],[0,0,1]]},
          "extrinsics": {"T_three": [[...4x4...]]}
        }
        """
         # Allow multiplexing to gripper tip handler
        data = request.get_json(force=True, silent=True) or {}
        typ = (data.get("type") or "").strip().lower()
        if typ == "gripper_tips":
            calib = data.get("gripper_tip_calib") or {}
            # minimal validation
            if not isinstance(calib, dict) or "left" not in calib or "right" not in calib:
                return jsonify({"error": "gripper_tip_calib must include 'left' and 'right' {x,y,z}"}), 400
            try:
                out_path = crowd_interface.save_gripper_tip_calibration(calib)
                return jsonify({"status": "ok", "path": out_path})
            except (ValueError, IOError) as e:
                return jsonify({"error": str(e)}), 400
        cam = data.get("camera")
        intr = data.get("intrinsics") or {}
        extr = data.get("extrinsics") or {}
        if not cam:
            return jsonify({"error": "missing 'camera'"}), 400
        if "Knew" not in intr or "width" not in intr or "height" not in intr:
            return jsonify({"error": "intrinsics must include width, height, Knew"}), 400
        if "T_three" not in extr:
            return jsonify({"error": "extrinsics must include T_three (4x4)"}), 400

        # Resolve ../calib path relative to this file
        base_dir = Path(__file__).resolve().parent
        calib_dir = (base_dir / ".." / "calib").resolve()
        calib_dir.mkdir(parents=True, exist_ok=True)
        out_path = calib_dir / f"manual_calibration_{cam}.json"

        # Write JSON file
        to_write = {
            "camera": cam,
            "intrinsics": {
                "width":  int(intr["width"]),
                "height": int(intr["height"]),
                "Knew":   intr["Knew"],
            },
            "extrinsics": {
                "T_three": extr["T_three"]
            }
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(to_write, f, indent=2)
        except Exception as e:
            return jsonify({"error": f"failed to write calibration: {e}"}), 500

        # Update in-memory models so the next /api/get-state reflects it immediately
        try:
            # intrinsics
            crowd_interface._camera_models[cam] = {
                "model": "pinhole",
                "rectified": crowd_interface._camera_models.get(cam, {}).get("rectified", False),
                "width":  int(intr["width"]),
                "height": int(intr["height"]),
                "Knew":   intr["Knew"],
            }
            # extrinsics (pose)
            crowd_interface._camera_poses[f"{cam}_pose"] = extr["T_three"]
        except Exception:
            # Non-fatal; file already saved
            pass

        return jsonify({"status": "ok", "path": str(out_path)})
        
    @app.route("/api/save-gripper-tips", methods=["POST"])
    def save_gripper_tips():
        try:
            data = request.get_json(force=True, silent=True) or {}
            calib = data.get("gripper_tip_calib") or {}
            if not isinstance(calib, dict) or "left" not in calib or "right" not in calib:
                return jsonify({"error": "gripper_tip_calib must include 'left' and 'right' {x,y,z}"}), 400
            out_path = crowd_interface.save_gripper_tip_calibration(calib)
            return jsonify({"status": "ok", "path": out_path})
        except (ValueError, IOError) as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"unexpected error: {e}"}), 500
    
    @app.route("/api/demo-videos/<filename>")
    def serve_demo_video(filename):
        """Serve demo video files for the frontend example video feature."""
        if not crowd_interface._demo_videos_dir:
            return jsonify({"error": "Demo videos directory not configured"}), 404
        
        try:
            # Sanitize filename to prevent directory traversal
            filename = os.path.basename(filename)
            file_path = crowd_interface._demo_videos_dir / filename
            
            if not file_path.exists():
                return jsonify({"error": "Video file not found"}), 404
            
            # Determine MIME type
            mime_type = mimetypes.guess_type(str(file_path))[0] or 'video/webm'
            
            # Create response with proper headers for video streaming
            response = make_response()
            response.headers['Content-Type'] = mime_type
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'public, max-age=3600'  # Cache for 1 hour
            
            # Read and return the file
            with open(file_path, 'rb') as f:
                response.data = f.read()
            
            return response
            
        except Exception as e:
            print(f"❌ Error serving demo video {filename}: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/show-videos/<video_id>")
    def serve_show_video(video_id):
        """
        Serve read-only example videos by numeric id from prompts/{task-name}/videos (or custom dir).
        This endpoint is independent of the recording feature and supports HTTP Range.
        """

        if not crowd_interface.show_demo_videos or not crowd_interface._show_videos_dir:
            return jsonify({"error": "Show demo videos is not enabled"}), 404

        # Sanitize; we only accept digits for ids.
        vid = "".join(c for c in str(video_id) if c.isdigit())
        if not vid:
            return jsonify({"error": "Invalid video id"}), 400

        file_path, mime = crowd_interface.find_show_video_by_id(vid)
        if not file_path:
            return jsonify({"error": "Video file not found"}), 404

        try:
            file_size = os.path.getsize(file_path)
            range_header = request.headers.get("Range", None)

            if range_header:
                # Format: "bytes=start-end"
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end or start >= file_size:
                        # RFC 7233
                        resp = Response(status=416)
                        resp.headers["Content-Range"] = f"bytes */{file_size}"
                        return resp

                    length = end - start + 1
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        data = f.read(length)

                    rv = Response(data, 206, mimetype=mime, direct_passthrough=True)
                    rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                    rv.headers["Accept-Ranges"] = "bytes"
                    rv.headers["Content-Length"] = str(length)
                    rv.headers["Cache-Control"] = "public, max-age=3600"
                    return rv

            # No Range: return full file
            with open(file_path, "rb") as f:
                data = f.read()
            rv = make_response(data)
            rv.headers["Content-Type"] = mime
            rv.headers["Content-Length"] = str(file_size)
            rv.headers["Accept-Ranges"] = "bytes"
            rv.headers["Cache-Control"] = "public, max-age=3600"
            return rv

        except Exception as e:
            print(f"❌ Error serving show video {video_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/show-videos/latest.webm")
    def serve_latest_show_video():
        """
        Serve the most recent .webm in the show_videos_dir with full HTTP Range support.
        Content-Type: video/webm
        Accept-Ranges: bytes
        """

        if not crowd_interface.show_demo_videos or not crowd_interface._show_videos_dir:
            return jsonify({"error": "Show demo videos is not enabled"}), 404

        # Resolve the latest numeric .webm (e.g., 1.webm, 2.webm, ...)
        latest_path, latest_id = crowd_interface.find_latest_show_video()
        if not latest_path or not latest_path.exists():
            return jsonify({"error": "No video file found"}), 404

        try:
            file_path = latest_path
            mime = "video/webm"  # force WebM for the player

            file_size = os.path.getsize(file_path)
            range_header = request.headers.get("Range", None)

            if range_header:
                # Format: "bytes=start-end"
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end or start >= file_size:
                        resp = Response(status=416)
                        resp.headers["Content-Range"] = f"bytes */{file_size}"
                        return resp

                    length = end - start + 1
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        data = f.read(length)

                    rv = Response(data, 206, mimetype=mime, direct_passthrough=True)
                    rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                    rv.headers["Accept-Ranges"] = "bytes"
                    rv.headers["Content-Length"] = str(length)
                    rv.headers["Cache-Control"] = "public, max-age=3600"
                    return rv

            # No Range header → return the whole file
            with open(file_path, "rb") as f:
                data = f.read()
            rv = make_response(data)
            rv.headers["Content-Type"] = mime
            rv.headers["Content-Length"] = str(file_size)
            rv.headers["Accept-Ranges"] = "bytes"
            rv.headers["Cache-Control"] = "public, max-age=3600"
            return rv

        except Exception as e:
            print(f"❌ Error serving latest show video: {e}")
            return jsonify({"error": str(e)}), 500

    # Streaming recording endpoints for canvas-based recording
    # Simple single recording session (no multi-user support needed)
    current_recording = None  # {recording_id, task_name, ext, chunks: [(seq, bytes)], started_at, metadata}
    
    @app.route("/api/record/start", methods=["POST"])
    def record_start():
        nonlocal current_recording
        
        
        if not crowd_interface.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400
        
        try:
            data = request.get_json() or {}
            recording_id = data.get('recording_id')
            task_name = data.get('task_name') or crowd_interface.task_name() or 'default'
            ext = 'webm'   # VP9-only
            
            if not recording_id:
                return jsonify({"error": "missing recording_id"}), 400
            
            # Initialize single recording session
            current_recording = {
                'recording_id': recording_id,
                'task_name': task_name,
                'ext': ext,
                'chunks': [],
                'started_at': data.get('started_at'),
                'metadata': data
            }
            
            return jsonify({"ok": True})
            
        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/record/chunk", methods=["POST"])
    def record_chunk():
        nonlocal current_recording
        
        
        if not crowd_interface.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400
        
        try:
            recording_id = request.args.get('rid')
            seq = request.args.get('seq', '0')
            
            if not current_recording:
                return jsonify({"error": "no active recording"}), 404
            
            if recording_id != current_recording['recording_id']:
                return jsonify({"error": "mismatched recording_id"}), 400
            
            # Get the raw bytes from the request
            chunk_data = request.get_data()
            if not chunk_data:
                return jsonify({"error": "no data"}), 400
            
            # Store chunk in memory (ordered by sequence)
            current_recording['chunks'].append((int(seq), chunk_data))
            
            return jsonify({"ok": True})
            
        except Exception as e:
            print(f"❌ Error storing chunk: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/record/stop", methods=["POST"])
    def record_stop():
        nonlocal current_recording
        
        if not crowd_interface.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400
        
        try:
            data = request.get_json() or {}
            recording_id = data.get('recording_id')
            
            if not current_recording:
                return jsonify({"error": "no active recording"}), 404
            
            if recording_id != current_recording['recording_id']:
                return jsonify({"error": "mismatched recording_id"}), 400
            
            # Sort chunks by sequence number
            chunks = sorted(current_recording['chunks'], key=lambda x: x[0])
            
            if not chunks:
                current_recording = None
                return jsonify({"error": "no chunks received"}), 400
            
            # Combine all chunks into a single video file
            try:
                # Get next filename using the counter system
                ext = current_recording['ext']
                filename, index = crowd_interface.next_video_filename(ext)
                file_path = crowd_interface._demo_videos_dir / filename
                
                # Write all chunks to the file
                with open(file_path, 'wb') as f:
                    for seq, chunk_data in chunks:
                        f.write(chunk_data)
                
                # Clean up current recording
                current_recording = None
                
                # No cloud upload - storing locally only
                public_url = None
                
                return jsonify({
                    "ok": True,
                    "filename": filename,
                    "path": str(file_path),
                    "save_dir_rel": crowd_interface.rel_path_from_repo(file_path.parent),
                    "public_url": public_url,
                    "index": index
                })
                
            except Exception as e:
                print(f"❌ Error finalizing recording: {e}")
                current_recording = None
                return jsonify({"error": f"failed to save: {e}"}), 500
            
        except Exception as e:
            print(f"❌ Error stopping recording: {e}")
            return jsonify({"error": str(e)}), 500
        
    @app.route("/api/record/save", methods=["POST"])
    def record_save():
        nonlocal current_recording
        
        """Manual save endpoint for demo video recordings"""
        
        if not crowd_interface.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400
        
        try:
            data = request.get_json() or {}
            recording_id = data.get('recording_id')
            
            if not current_recording:
                return jsonify({"error": "No active recording session found"}), 404
            
            if recording_id != current_recording['recording_id']:
                return jsonify({"error": "Recording ID mismatch"}), 400
            
            # Sort chunks by sequence number
            chunks = sorted(current_recording['chunks'], key=lambda x: x[0])
            
            if not chunks:
                current_recording = None
                return jsonify({"error": "No recording data to save"}), 400
            
            # Get next filename using the counter system
            ext = current_recording['ext']
            filename, index = crowd_interface.next_video_filename(ext)
            file_path = crowd_interface._demo_videos_dir / filename
            
            # Write all chunks to the file
            with open(file_path, 'wb') as f:
                for seq, chunk_data in chunks:
                    f.write(chunk_data)
            
            # Clean up current recording
            current_recording = None
            
            # No cloud upload - storing locally only
            public_url = None
            
            return jsonify({
                "ok": True,
                "status": "success",
                "message": "Recording saved successfully",
                "filename": filename,
                "path": str(file_path),
                "save_dir_rel": crowd_interface.rel_path_from_repo(file_path.parent),
                "public_url": public_url,
                "index": index
            })
            
        except Exception as e:
            print(f"❌ Error saving recording: {e}")
            return jsonify({"error": str(e)}), 500
    
    # ============================================================================
    # Animation API Endpoints
    # ============================================================================
    
    def get_session_id():
        """Extract session ID from request headers"""
        return request.headers.get('X-Session-ID', 'anonymous')
    
    @app.route("/api/animation/status", methods=["GET"])
    def animation_status():
        """Get animation slot availability and current status"""
        try:
            status = crowd_interface.get_animation_status()
            return jsonify(status)
        except Exception as e:
            print(f"❌ Error getting animation status: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/animation/start", methods=["POST"])
    def start_animation():
        """Start animation for the current user session"""
        try:
            session_id = get_session_id()
            data = request.get_json() or {}
            
            # Extract goal pose and animation parameters
            goal_pose = data.get('goal_pose')
            goal_joints = data.get('goal_joints')
            duration = data.get('duration', 3.0)
            gripper_action = data.get('gripper_action')  # NEW: extract gripper action
            
            # Validate input
            if not goal_pose and not goal_joints:
                return jsonify({"error": "Must provide either goal_pose or goal_joints"}), 400
            
            result = crowd_interface.start_animation(
                session_id=session_id,
                goal_pose=goal_pose,
                goal_joints=goal_joints,
                duration=duration,
                gripper_action=gripper_action
            )
            
            if result.get("status") == "error":
                return jsonify(result), 400
            
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Error starting animation: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/animation/stop", methods=["POST"])
    def stop_animation():
        """Stop animation for the current user session"""
        try:
            session_id = get_session_id()
            
            result = crowd_interface.stop_animation(session_id)
            
            if result.get("status") == "error":
                return jsonify(result), 400
                
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Error stopping animation: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/animation/frame", methods=["GET"])
    def capture_animation_frame():
        """Capture current animation frame for the user session"""
        try:
            session_id = get_session_id()
            
            result = crowd_interface.capture_animation_frame(session_id)
            
            if result.get("status") == "error":
                return jsonify(result), 400
                
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Error capturing animation frame: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/animation/release", methods=["POST"])
    def release_animation_session():
        """Release animation slot for disconnected session"""
        try:
            session_id = get_session_id()
            
            result = crowd_interface.release_animation_session(session_id)
            
            return jsonify(result)
            
        except Exception as e:
            print(f"❌ Error releasing animation session: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/simulation/init", methods=["POST"])
    def init_simulation():
        """Initialize simulation mode by triggering initial state capture"""
        try:
            if hasattr(crowd_interface, 'isaac_manager') and crowd_interface.isaac_manager:
                # Create a basic config to trigger simulation initialization
                basic_config = {
                    "usd_path": f"public/assets/usd/{crowd_interface.task_name}_flattened.usd",
                    "robot_joints": [0.0] * 7,  # Default joint positions
                    "object_poses": {
                        "Cube_Blue": {"pos": [0.2, 0.0, 0.1], "rot": [0, 0, 0, 1]},
                        "Cube_Red": {"pos": [0.2, 0.2, 0.1], "rot": [0, 0, 0, 1]},
                        "Tennis": {"pos": [0.2, -0.2, 0.1], "rot": [0, 0, 0, 1]}
                    }
                }
                # Use the actual method that exists
                result = crowd_interface.isaac_manager.capture_initial_state(basic_config)
                if isinstance(result, dict) and "status" in result:
                    return jsonify(result)
                else:
                    # capture_initial_state returns file paths, not status
                    return jsonify({"status": "success", "message": "Simulation initialized", "files": result})
            else:
                return jsonify({"status": "error", "message": "Isaac manager not available"}), 400
        except Exception as e:
            print(f"❌ Error initializing simulation: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route("/api/simulation/status", methods=["GET"])
    def simulation_status():
        """Get simulation status"""
        try:
            if hasattr(crowd_interface, 'isaac_manager') and crowd_interface.isaac_manager:
                status = {
                    "simulation_initialized": crowd_interface.isaac_manager.simulation_initialized,
                    "worker_ready": crowd_interface.isaac_manager.worker_ready,
                    "animation_initialized": crowd_interface.isaac_manager.animation_initialized
                }
                return jsonify(status)
            else:
                return jsonify({"status": "error", "message": "Isaac manager not available"}), 400
        except Exception as e:
            print(f"❌ Error getting simulation status: {e}")
            return jsonify({"error": str(e)}), 500
    
    return app