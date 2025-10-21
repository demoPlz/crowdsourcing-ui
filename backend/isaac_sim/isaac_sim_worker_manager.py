#!/usr/bin/env python3
"""
Backend integration for persistent Isaac Sim worker
Manages long-running Isaac Sim subprocess for state cycling
"""

import os
import json
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, Optional, Any

class PersistentWorkerManager:
    """Manages a long-running Isaac Sim worker subprocess for state cycling"""
    
    def __init__(self, isaac_sim_path: str, output_base_dir: str = "/tmp/isaac_worker", max_animation_users: int = 2):
        self.isaac_sim_path = isaac_sim_path  # Path to Isaac Sim installation
        self.output_base_dir = output_base_dir
        self.max_animation_users = max_animation_users
        self.worker_process = None
        self.worker_ready = False
        self.simulation_initialized = False
        self.animation_initialized = False
        
        # Animation user management
        self.animation_users = {}  # user_id -> {'session_id': str, 'active': bool, 'start_time': float}
        self.available_slots = set(range(max_animation_users))  # Available animation environment slots
        self.session_to_user = {}  # session_id -> user_id mapping
        
        # Communication files
        self.command_file = f"{output_base_dir}/commands.json"
        self.result_file = f"{output_base_dir}/results.json"
        self.status_file = f"{output_base_dir}/status.json"
        
        os.makedirs(output_base_dir, exist_ok=True)
        
    def start_worker(self, initial_config: Dict[str, Any], max_users: int = 10):
        """Start the persistent Isaac Sim worker process"""
        if self.worker_process and self.worker_process.poll() is None:
            print("Worker already running")
            return
            
        print("Starting persistent Isaac Sim worker...")
        
        # Save initial config
        config_file = f"{self.output_base_dir}/initial_config.json"
        with open(config_file, 'w') as f:
            json.dump(initial_config, f)
            
        # Start worker in Isaac Sim environment
        worker_script = os.path.join(os.path.dirname(__file__), "persistent_isaac_sim_worker.py")
        cmd = [
            f"{self.isaac_sim_path}/python.sh",  # Isaac Sim's Python
            worker_script,
            "--config", config_file,
            "--output-dir", self.output_base_dir,
            "--max-users", str(max_users)
        ]
        
        # Create clean environment for Isaac Sim (remove conda variables)
        clean_env = os.environ.copy()
        
        # Remove conda-related environment variables
        conda_vars_to_remove = [
            'CONDA_DEFAULT_ENV',
            'CONDA_EXE', 
            'CONDA_PREFIX',
            'CONDA_PROMPT_MODIFIER',
            'CONDA_PYTHON_EXE',
            'CONDA_SHLVL',
            '_CE_CONDA',
            '_CE_M'
        ]
        
        for var in conda_vars_to_remove:
            clean_env.pop(var, None)
            
        # Also clean PATH to remove conda paths
        if 'PATH' in clean_env:
            path_parts = clean_env['PATH'].split(':')
            # Remove paths containing 'conda' or 'miniconda' or 'anaconda'
            clean_path_parts = [p for p in path_parts 
                              if not any(x in p.lower() for x in ['conda', 'miniconda', 'anaconda'])]
            clean_env['PATH'] = ':'.join(clean_path_parts)
        
        print(f"Starting Isaac Sim worker with clean environment...")
        print(f"Command: {' '.join(cmd)}")
        
        self.worker_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=clean_env  # Use clean environment
        )
        
        # Monitor worker output
        self._start_output_monitor()
        
        # Wait for worker to be ready
        self._wait_for_ready()
        
        # Auto-initialize animation mode after first state capture (pre-clone strategy)
        # This will be triggered in capture_initial_state()
        
    def _start_output_monitor(self):
        """Monitor worker output in separate thread"""
        def monitor_stdout():
            for line in iter(self.worker_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    print(f"[Worker] {line}")
                    # Check for ready signals
                    if "WORKER_READY" in line:
                        self.worker_ready = True
                    elif "SIMULATION_INITIALIZED" in line:
                        self.simulation_initialized = True
                        
        def monitor_stderr():
            for line in iter(self.worker_process.stderr.readline, ''):
                line = line.strip()
                if line:
                    print(f"[Worker Error] {line}")
                    
        threading.Thread(target=monitor_stdout, daemon=True).start()
        threading.Thread(target=monitor_stderr, daemon=True).start()
        
    def _wait_for_ready(self, timeout: int = 60):
        """Wait for worker to signal ready"""
        start_time = time.time()
        while not self.worker_ready and (time.time() - start_time) < timeout:
            if self.worker_process.poll() is not None:
                raise RuntimeError("Worker process terminated unexpectedly")
            time.sleep(0.1)
            
        if not self.worker_ready:
            raise TimeoutError("Worker did not become ready within timeout")
            
        print("✓ Isaac Sim worker is ready for commands")

    def _verify_worker_simulation_ready(self, timeout: int = 10) -> bool:
        """Explicitly verify that worker's internal simulation state is ready"""
        for attempt in range(timeout):
            try:
                # Query worker's actual internal state
                status_command = {"action": "get_status"}
                status_result = self._send_command(status_command)
                
                # Check if worker reports simulation as initialized
                worker_sim_ready = status_result.get("simulation_initialized", False)
                
                if worker_sim_ready:
                    print("✓ Worker simulation state verified as ready")
                    return True
                    
                print(f"Worker simulation not ready yet (attempt {attempt + 1}/{timeout})")
                time.sleep(1)  # Wait 500ms before retry
                
            except Exception as e:
                print(f"Status check failed (attempt {attempt + 1}): {e}")
                time.sleep(0.5)
                
        print(f"⚠ Worker simulation verification timed out after {timeout} attempts")
        return False
        
    def capture_initial_state(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Initialize simulation and capture first state"""
        if not self.worker_ready:
            raise RuntimeError("Worker not ready. Call start_worker() first.")
            
        output_dir = f"{self.output_base_dir}/initial_state"
        
        command = {
            "action": "initialize_and_capture",
            "config": config,
            "output_dir": output_dir
        }
        
        result = self._send_command(command)
        if result.get("status") == "success":
            self.simulation_initialized = True

            print("Verifying worker simulation state before initializing animation...")
            if not self._verify_worker_simulation_ready():
                print("⚠ Worker simulation verification failed, skipping animation initialization")
                return result
            
            # Auto-initialize animation mode with pre-cloning
            print(f"Auto-initializing animation mode with {self.max_animation_users} environments...")
            try:
                anim_command = {
                    "action": "initialize_animation",
                    "max_users": self.max_animation_users
                }
                anim_result = self._send_command(anim_command)
                print(f"Animation initialization result: {anim_result}")
                if anim_result.get("status") == "success":
                    self.animation_initialized = True
                    print("✓ Animation environments pre-cloned and ready")
                else:
                    print(f"⚠ Animation initialization failed: {anim_result.get('message', 'Unknown error')}")
                    # Don't fail the whole system if animation fails, just disable it
                    self.animation_initialized = False
            except Exception as e:
                print(f"⚠ Failed to initialize animation mode: {e}")
                import traceback
                traceback.print_exc()
                self.animation_initialized = False
                
        return result
        
    def start_user_animation(self, user_id: int, goal_pose: Dict = None, 
                           goal_joints: list = None, duration: float = 3.0, gripper_action: str = None) -> Dict[str, Any]:
        """Start animation for specific user"""
        if not self.animation_initialized:
            return {"status": "error", "message": "Animation not initialized"}
            
        command = {
            "action": "start_user_animation",
            "user_id": user_id,
            "goal_pose": goal_pose,
            "goal_joints": goal_joints,
            "duration": duration,
            "gripper_action": gripper_action
        }
        
        return self._send_command(command)
        
    def stop_user_animation(self, user_id: int) -> Dict[str, Any]:
        """Stop animation for specific user"""
        command = {
            "action": "stop_user_animation",
            "user_id": user_id
        }
        
        return self._send_command(command)
        
    def update_state_and_sync_animations(self, config: Dict[str, Any], state_id: str = None) -> Dict[str, str]:
        """Update simulation state and synchronize all animation environments"""
        if not self.worker_ready:
            raise RuntimeError("Worker not ready. Call start_worker() first.")
            
        state_id = state_id or f"state_{int(time.time())}"
        output_dir = f"{self.output_base_dir}/{state_id}"
        
        # First update the base state and capture static images
        command = {
            "action": "update_and_capture",
            "config": config,
            "output_dir": output_dir,
            "state_id": state_id
        }
        
        result = self._send_command(command)
        
        # Then synchronize all animation environments to the new state
        if self.animation_initialized and result.get("status") == "success":
            try:
                sync_command = {
                    "action": "sync_animation_environments",
                    "config": config
                }
                sync_result = self._send_command(sync_command)
                if sync_result.get("status") != "success":
                    print(f"Warning: Failed to sync animation environments: {sync_result.get('message', 'Unknown error')}")
            except Exception as e:
                print(f"Warning: Failed to sync animation environments: {e}")
                
        return result
    
    def start_user_animation_managed(self, session_id: str, goal_pose: Dict = None, 
                                   goal_joints: list = None, duration: float = 3.0, gripper_action: str = None) -> Dict[str, Any]:
        """Start animation for a session (manages user slots automatically)"""
        if not self.animation_initialized:
            return {"status": "error", "message": "Animation not initialized"}
            
        # Check if session already has a slot
        if session_id in self.session_to_user:
            user_id = self.session_to_user[session_id]
            if user_id in self.animation_users and self.animation_users[user_id]['active']:
                return {"status": "error", "message": "Session already has active animation"}
        
        # Allocate a new slot
        if not self.available_slots:
            return {"status": "error", "message": "No animation slots available"}
            
        user_id = self.available_slots.pop()
        
        # Start animation
        result = self.start_user_animation(user_id, goal_pose, goal_joints, duration, gripper_action)
        
        if result.get("status") == "success":
            # Track session -> user mapping
            self.session_to_user[session_id] = user_id
            self.animation_users[user_id] = {
                'session_id': session_id,
                'active': True,
                'start_time': time.time()
            }
        else:
            # Return slot if animation failed
            self.available_slots.add(user_id)
            
        return result
    
    def stop_user_animation_managed(self, session_id: str) -> Dict[str, Any]:
        """Stop animation for a session"""
        if session_id not in self.session_to_user:
            return {"status": "error", "message": "Session not found"}
            
        user_id = self.session_to_user[session_id]
        result = self.stop_user_animation(user_id)
        
        # Release slot
        if user_id in self.animation_users:
            del self.animation_users[user_id]
        if session_id in self.session_to_user:
            del self.session_to_user[session_id]
        self.available_slots.add(user_id)
        
        return result
    
    def get_user_by_session(self, session_id: str) -> int | None:
        """Get user_id for a session"""
        return self.session_to_user.get(session_id)
    
    def capture_user_frame(self, user_id: int) -> Dict[str, Any]:
        """Capture frame for specific user"""
        if not self.animation_initialized:
            return {"status": "error", "message": "Animation not initialized"}
            
        command = {
            "action": "capture_user_frame",
            "user_id": user_id,
            "output_dir": f"{self.output_base_dir}/user_{user_id}_frame"
        }
        
        return self._send_command(command)
    
    def release_animation_slot(self, user_id: int) -> bool:
        """Release animation slot for a user"""
        if user_id in self.animation_users:
            session_id = self.animation_users[user_id].get('session_id')
            if session_id and session_id in self.session_to_user:
                del self.session_to_user[session_id]
            del self.animation_users[user_id]
        
        self.available_slots.add(user_id)
        return True
    
    def get_animation_status(self) -> Dict[str, Any]:
        """Get comprehensive animation status"""
        active_users = len([u for u in self.animation_users.values() if u.get('active')])
        
        return {
            "animation_initialized": self.animation_initialized,
            "max_users": self.max_animation_users,
            "available_slots": len(self.available_slots),
            "active_users": active_users,
            "users": {
                session_id: {
                    "user_id": user_id,
                    "active": self.animation_users[user_id].get('active', False),
                    "start_time": self.animation_users[user_id].get('start_time', 0)
                }
                for session_id, user_id in self.session_to_user.items()
                if user_id in self.animation_users
            }
        }
    
    def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to worker and wait for result"""
        # Write command to file
        with open(self.command_file, 'w') as f:
            json.dump(command, f)
            
        # Signal worker (it polls the command file)
        command_signal_file = f"{self.command_file}.signal"
        Path(command_signal_file).touch()
        
        # Wait for result
        result_signal_file = f"{self.result_file}.signal"
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        while not os.path.exists(result_signal_file):
            if time.time() - start_time > timeout:
                raise TimeoutError("Command timeout")
            time.sleep(0.1)
            
        # Read result
        with open(self.result_file, 'r') as f:
            result = json.load(f)
            
        # Cleanup signal files
        if os.path.exists(command_signal_file):
            os.remove(command_signal_file)
        if os.path.exists(result_signal_file):
            os.remove(result_signal_file)
            
        return result
        
    def get_worker_status(self) -> Dict[str, Any]:
        """Get current worker status"""
        if not self.worker_process:
            return {"status": "not_started"}
            
        if self.worker_process.poll() is not None:
            return {"status": "terminated", "return_code": self.worker_process.returncode}
            
        return {
            "status": "running",
            "ready": self.worker_ready,
            "simulation_initialized": self.simulation_initialized,
            "pid": self.worker_process.pid
        }
        
    def stop_worker(self):
        """Stop the worker process gracefully"""
        if not self.worker_process:
            return
            
        print("Stopping Isaac Sim worker...")
        
        try:
            # Try graceful shutdown first
            command = {"action": "shutdown"}
            self._send_command(command)
            
            # Wait a bit for graceful shutdown
            self.worker_process.wait(timeout=10)
        except (TimeoutError, subprocess.TimeoutExpired):
            print("Graceful shutdown failed, terminating...")
            self.worker_process.terminate()
            try:
                self.worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Force killing worker...")
                self.worker_process.kill()
                
        self.worker_process = None
        self.worker_ready = False
        self.simulation_initialized = False
        print("✓ Worker stopped")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_worker()

# Backend usage example
class IsaacSimBackend:
    """Clean backend integration for dual-purpose Isaac Sim worker"""
    
    def __init__(self, isaac_sim_path: str):
        self.isaac_sim_path = isaac_sim_path
        self.worker_manager = None
        
    def initialize(self, initial_config: Dict[str, Any]):
        """Initialize the Isaac Sim backend with dual-purpose functionality"""
        self.worker_manager = PersistentWorkerManager(self.isaac_sim_path)
        self.worker_manager.start_worker(initial_config)
        
        # Capture initial state (auto-initializes animation mode)
        return self.worker_manager.capture_initial_state(initial_config)
        
    def update_state(self, config: Dict[str, Any], state_id: str = None) -> Dict[str, str]:
        """Update state with dual-purpose sync (static images + animation environments)"""
        if not self.worker_manager:
            raise RuntimeError("Backend not initialized")
        return self.worker_manager.update_state_and_sync_animations(config, state_id)
        
    def start_user_animation(self, user_id: int, goal_pose: Dict = None, 
                           goal_joints: list = None, duration: float = 3.0) -> Dict[str, Any]:
        """Start animation for user"""
        if not self.worker_manager:
            raise RuntimeError("Backend not initialized")
        return self.worker_manager.start_user_animation(user_id, goal_pose, goal_joints, duration)
        
    def stop_user_animation(self, user_id: int) -> Dict[str, Any]:
        """Stop animation for user"""
        if not self.worker_manager:
            raise RuntimeError("Backend not initialized")
        return self.worker_manager.stop_user_animation(user_id)
        
    def shutdown(self):
        """Shutdown the backend"""
        if self.worker_manager:
            self.worker_manager.stop_worker()
            self.worker_manager = None