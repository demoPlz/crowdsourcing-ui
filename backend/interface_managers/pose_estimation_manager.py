"""
Pose Estimation Manager Module

Handles 6D pose estimation for objects using any6d workers.
Manages cross-environment communication via disk-based job queues.
"""

import os
import time
import json
import uuid
import subprocess
from pathlib import Path
from threading import Thread, Lock

import numpy as np


class PoseEstimationManager:
    """
    Manages 6D pose estimation for objects using any6d workers.
    
    Responsibilities:
    - Spawn and manage any6d worker processes (one per object)
    - Disk-based job queue management (inbox/outbox)
    - Background results watcher thread
    - Intrinsics selection and job creation
    - Results integration into state info
    
    Attributes:
        pose_jobs_root: Root directory for job queues
        pose_inbox: Directory where jobs are enqueued
        pose_outbox: Directory where results are written
        pose_tmp: Temporary directory for atomic writes
    """
    
    def __init__(
        self,
        obs_cache_root: Path,
        object_mesh_paths: dict[str, str] | None,
        objects: dict[str, str] | None,
        calibration_manager,
        state_lock: Lock,
        pending_states_by_episode: dict,
    ):
        """
        Initialize pose estimation manager.
        
        Args:
            obs_cache_root: Root directory for observation cache (parent of pose_jobs)
            object_mesh_paths: Dict of object name -> mesh file path
            objects: Dict of object name -> language prompt
            calibration_manager: CalibrationManager instance for intrinsics access
            state_lock: Lock protecting pending_states_by_episode
            pending_states_by_episode: Reference to episode state dict for result integration
        """
        self.object_mesh_paths = object_mesh_paths
        self.objects = objects
        self.calibration = calibration_manager
        self.state_lock = state_lock
        self.pending_states_by_episode = pending_states_by_episode
        
        # Disk-backed job queue shared with any6d env workers
        self.pose_jobs_root = (obs_cache_root / "pose_jobs").resolve()
        self.pose_inbox = self.pose_jobs_root / "inbox"
        self.pose_outbox = self.pose_jobs_root / "outbox"
        self.pose_tmp = self.pose_jobs_root / "tmp"
        
        # Create directories
        for d in (self.pose_inbox, self.pose_outbox, self.pose_tmp):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        
        # Clean up stale jobs from previous runs
        self._cleanup_job_queues()
        
        # Worker process management
        self._pose_worker_procs: dict[str, subprocess.Popen] = {}
        self._pose_results_thread: Thread | None = None
        
        # Start workers and results watcher
        self._start_pose_workers()
        self._start_pose_results_watcher()
    
    def _cleanup_job_queues(self):
        """
        Remove stale job files from inbox and outbox directories.
        Called on initialization to prevent workers from processing jobs from previous runs.
        """
        try:
            # Clean inbox (pending jobs)
            for f in self.pose_inbox.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass
            
            # Clean outbox (completed results)
            for f in self.pose_outbox.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass
            
            # Clean tmp (partially written jobs)
            for f in self.pose_tmp.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass
            
            print("🧹 Cleaned up stale pose jobs from previous runs")
        except Exception as e:
            print(f"⚠️  Failed to cleanup job queues: {e}")
    
    def _start_pose_workers(self):
        """
        Spawn ONE persistent worker per object (they run continuously and process jobs sequentially).
        Worker script path can be overridden via $POSE_WORKER_SCRIPT.
        
        Set SKIP_POSE_WORKERS=1 to disable auto-spawning (useful for manual debugging).
        """
        if os.getenv("SKIP_POSE_WORKERS", "0") == "1":
            print("🐛 SKIP_POSE_WORKERS=1: Not spawning pose workers (attach manually)")
            return
            
        if not self.object_mesh_paths:
            print("⚠️  No object_mesh_paths provided; pose workers not started.")
            return
        
        worker_script = os.getenv(
            "POSE_WORKER_SCRIPT",
            str((Path(__file__).resolve().parent / "any6d" / "pose_worker.py").resolve())
        )
        pose_env = os.getenv("POSE_ENV", "any6d")
        
        # Build CUDA library paths for any6d
        conda_prefix = Path.home() / "miniconda3" / "envs" / pose_env
        cuda_lib_path = f"{conda_prefix}/lib:{conda_prefix}/targets/x86_64-linux/lib"
        worker_env = os.environ.copy()
        worker_env["LD_LIBRARY_PATH"] = cuda_lib_path
        
        # Spawn ONE persistent worker per object (parallel processing)
        print("🔄 Starting pose estimation workers (one per object)...")
        
        # Track ready status for each worker (True=ready, False=pending, None=failed)
        workers_status = {obj: False for obj in self.object_mesh_paths.keys()}
        status_lock = Lock()
        
        for obj, mesh_path in self.object_mesh_paths.items():
            lang_prompt = (self.objects or {}).get(obj, obj)
            
            cmd = [
                "conda", "run", "--no-capture-output", "-n", pose_env,
                "python", worker_script,
                "--jobs-dir", str(self.pose_jobs_root),
                "--object", obj,
                "--mesh", str(mesh_path),
                "--prompt", str(lang_prompt)
            ]
            
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=worker_env
                )
                self._pose_worker_procs[obj] = proc
                print(f"✓ Pose worker for '{obj}' started (PID {proc.pid})")
                
                # Start thread to print worker output and detect ready/failure signals
                def _print_worker_output(proc, obj_name):
                    try:
                        for line in iter(proc.stdout.readline, ""):
                            if line:
                                print(f"[{obj_name}] {line.rstrip()}")
                                # Detect ready signal: "✅ worker ready (watching ...)"
                                if "✅" in line and "worker ready" in line:
                                    with status_lock:
                                        workers_status[obj_name] = True
                                # Detect initialization failures: "✖ mesh load failed" or "✖ engines init failed"
                                elif "✖" in line and ("mesh load failed" in line or "engines init failed" in line):
                                    with status_lock:
                                        workers_status[obj_name] = None  # None indicates failure
                    except Exception:
                        pass
                    finally:
                        proc.stdout.close()
                Thread(target=_print_worker_output, args=(proc, obj), daemon=True).start()
                
            except Exception as e:
                print(f"⚠️  Failed to start pose worker for '{obj}': {e}")
                with status_lock:
                    workers_status[obj] = None  # Mark as failed
        
        # Wait for all workers to be ready or fail (with timeout)
        print("⏳ Waiting for pose workers to initialize...")
        timeout_s = float(os.getenv("POSE_WORKER_INIT_TIMEOUT", "30.0"))
        deadline = time.time() + timeout_s
        
        while time.time() < deadline:
            with status_lock:
                # Check if any workers failed (None status)
                failed = [obj for obj, status in workers_status.items() if status is None]
                if failed:
                    print(f"❌ Worker initialization FAILED for: {failed}")
                    print(f"❌ Check worker logs above for details (mesh load or engine init errors)")
                    # Continue anyway - maybe other workers can still work
                    return
                
                # Check if all workers are ready (True status)
                if all(status is True for status in workers_status.values()):
                    print("✅ All pose workers ready!")
                    return
            time.sleep(0.1)
        
        # Timeout - report which workers are still pending
        with status_lock:
            pending = [obj for obj, status in workers_status.items() if status is False]
            failed = [obj for obj, status in workers_status.items() if status is None]
        
        if failed:
            print(f"❌ Workers failed during initialization: {failed}")
        if pending:
            print(f"⚠️  Timeout waiting for pose workers: {pending} not ready after {timeout_s}s")
        if pending or failed:
            print(f"⚠️  Continuing anyway, but pose estimation may fail for affected objects...")

    def _start_pose_results_watcher(self):
        self._pose_results_thread = Thread(target=self._pose_results_watcher, daemon=True)
        self._pose_results_thread.start()

    def _pose_results_watcher(self):
        """
        Poll pose_jobs/outbox for result JSONs and fold them into state_info['object_poses'].
        """
        print("📬 Results watcher thread started")
        while True:
            try:
                for p in self.pose_outbox.glob("*.json"):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            result = json.load(f)
                        os.remove(p)
                        
                        ep_id = result.get("episode_id")
                        st_id = result.get("state_id")
                        obj = result.get("object")
                        pose = result.get("pose_cam_T_obj")  # may be None on failure
                        
                        with self.state_lock:
                            ep = self.pending_states_by_episode.get(ep_id)
                            if not ep or st_id not in ep:
                                continue
                            st = ep[st_id]
                            if "object_poses" not in st:
                                st["object_poses"] = {}
                            st["object_poses"][obj] = pose
                    except Exception as e:
                        print(f"⚠️  Failed to process pose result {p.name}: {e}")
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            except Exception:
                # Keep the watcher alive
                time.sleep(0.2)
            time.sleep(0.1)

    def _intrinsics_for_pose(self) -> list[list[float]]:
        """
        Returns 3x3 K to send to pose workers.
        Returns a Python list-of-lists (JSON-serializable).
        """
        realsense_calib = self.calibration.repo_root / "data" / "calib" / "intrinsics_realsense_d455.npz"
        if realsense_calib.exists():
            data = np.load(realsense_calib, allow_pickle=True)
            K = np.asarray(data["Knew"], dtype=np.float64)  # Use Knew (same as K for RealSense)
            return K.tolist()
        else:
            print("⚠️  RealSense D455 intrinsics not found")
            exit(1)

    def enqueue_pose_jobs_for_state(
        self,
        episode_id: str,
        state_id: int,
        state_info: dict,
        wait: bool = True,
        timeout_s: float | None = None,
    ) -> bool:
        """
        Enqueue one pose-estimation job per object into pose_jobs/inbox, then
        (optionally) block until results for *all* objects are folded into
        pending_states_by_episode[episode_id][state_id]['object_poses'] by the
        results watcher.

        Returns:
            True  -> all objects reported (success or failure) within timeout
            False -> state disappeared or timed out before all objects reported
        """
        if not self.object_mesh_paths:
            # Nothing to do; treat as ready.
            return True

        expected_objs = list(self.object_mesh_paths.keys())

        # ---------- Enqueue jobs (do not mark object_poses yet) ----------
        print(f"📬 Enqueueing pose jobs for episode={episode_id} state={state_id}")
        for obj, mesh_path in self.object_mesh_paths.items():
            job_id = f"{episode_id}_{state_id}_{obj}_{uuid.uuid4().hex[:8]}"
            job = {
                "job_id": job_id,
                "episode_id": int(episode_id),
                "state_id": int(state_id),
                "object": obj,
                "obs_path": state_info.get("obs_path"),
                "K": self._intrinsics_for_pose(),             # 3x3 list
                "prompt": (self.objects or {}).get(obj, obj), # language prompt
                # Optional knobs:
                "est_refine_iter": int(os.getenv("POSE_EST_ITERS", "20")),
                "track_refine_iter": int(os.getenv("POSE_TRACK_ITERS", "8")),
            }
            print(f"   📝 Creating job {job_id}")
            print(f"      obj={obj}, obs_path={job['obs_path']}")
            tmp = self.pose_tmp / f"{job_id}.json"
            dst = self.pose_inbox / f"{job_id}.json"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(job, f)
                os.replace(tmp, dst)  # atomic move
                print(f"   ✅ Job written to inbox: {dst.name}")
            except Exception as e:
                print(f"⚠️  Failed to enqueue pose job {job_id}: {e}")

        if not wait:
            return True

        # ---------- Wait for watcher to fold ALL results into state ----------
        # NOTE: Do NOT hold self.state_lock while sleeping; watcher needs it.
        try:
            timeout = float(timeout_s if timeout_s is not None else os.getenv("POSE_WAIT_TIMEOUT_S", "20.0"))
        except Exception:
            timeout = 20.0
        deadline = time.time() + max(0.0, timeout)

        # We consider a job "done" when the watcher has inserted a key for that object,
        # regardless of success (pose may be None on failure). Presence == finished.
        while True:
            with self.state_lock:
                ep = self.pending_states_by_episode.get(episode_id)
                if not ep or state_id not in ep:
                    print(f"⚠️  State disappeared (ep={episode_id}, state={state_id})")
                    return False
                st = ep[state_id]
                poses = st.get("object_poses", {})
                done = all(obj in poses for obj in expected_objs)

            if done:
                return True

            # if time.time() > deadline:
            #     with self.state_lock:
            #         poses_now = list(self.pending_states_by_episode.get(episode_id, {}).get(state_id, {}).get("object_poses", {}).keys())
            #     print(f"⚠️  Timed out waiting for poses (ep={episode_id}, state={state_id}). "
            #         f"Have={poses_now}, expected={expected_objs}")
            #     return False

            time.sleep(0.02)
    
    def stop(self):
        """Stop all pose worker processes."""
        for obj, proc in self._pose_worker_procs.items():
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
                print(f"✓ Pose worker for '{obj}' stopped")
            except Exception as e:
                print(f"⚠️  Failed to stop pose worker for '{obj}': {e}")
        self._pose_worker_procs.clear()
