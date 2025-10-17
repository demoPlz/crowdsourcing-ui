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
import queue
from pathlib import Path
from typing import Dict, Optional, Any

class PersistentWorkerManager:
    """Manages a long-running Isaac Sim worker subprocess for state cycling"""
    
    def __init__(self, isaac_sim_path: str, output_base_dir: str = "/tmp/isaac_worker"):
        self.isaac_sim_path = isaac_sim_path  # Path to Isaac Sim installation
        self.output_base_dir = output_base_dir
        self.worker_process = None
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.worker_ready = False
        self.simulation_initialized = False
        
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
        
    def update_state_and_capture(self, config: Dict[str, Any], state_id: str = None) -> Dict[str, str]:
        """Update simulation state and capture images"""
        if not self.worker_ready:
            raise RuntimeError("Worker not ready. Call start_worker() first.")
            
        state_id = state_id or f"state_{int(time.time())}"
        output_dir = f"{self.output_base_dir}/{state_id}"
        
        # Send command to worker
        command = {
            "action": "update_and_capture",
            "config": config,
            "output_dir": output_dir,
            "state_id": state_id
        }
        
        return self._send_command(command)
        
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
        return result
        
    def initialize_animation_mode(self, max_users: int = 8) -> Dict[str, Any]:
        """Initialize animation mode with cloned environments"""
        if not self.simulation_initialized:
            raise RuntimeError("Must initialize simulation first. Call capture_initial_state().")
            
        command = {
            "action": "initialize_animation",
            "max_users": max_users
        }
        
        return self._send_command(command)
        
    def start_user_animation(self, user_id: int, goal_pose: Dict = None, 
                           goal_joints: list = None, duration: float = 3.0) -> Dict[str, Any]:
        """Start animation for specific user"""
        command = {
            "action": "start_user_animation",
            "user_id": user_id,
            "goal_pose": goal_pose,
            "goal_joints": goal_joints,
            "duration": duration
        }
        
        return self._send_command(command)
        
    def stop_user_animation(self, user_id: int) -> Dict[str, Any]:
        """Stop animation for specific user"""
        command = {
            "action": "stop_user_animation",
            "user_id": user_id
        }
        
        return self._send_command(command)
        
    def capture_user_frame(self, user_id: int, output_dir: str = None) -> Dict[str, Any]:
        """Capture frame for specific user"""
        output_dir = output_dir or f"{self.output_base_dir}/user_frames"
        
        command = {
            "action": "capture_user_frame",
            "user_id": user_id,
            "output_dir": output_dir
        }
        
        return self._send_command(command)
        
    def start_animation_loop(self, output_dir: str = None) -> Dict[str, Any]:
        """Start the main animation loop (blocking operation)"""
        output_dir = output_dir or f"{self.output_base_dir}/animation_frames"
        
        command = {
            "action": "start_animation_loop",
            "output_dir": output_dir
        }
        
        # Note: This will block until animation loop ends
        return self._send_command(command)
        
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
    """Example backend integration using the persistent worker"""
    
    def __init__(self, isaac_sim_path: str):
        self.isaac_sim_path = isaac_sim_path
        self.worker_manager = None
        
    def initialize(self, initial_config: Dict[str, Any]):
        """Initialize the Isaac Sim backend"""
        self.worker_manager = PersistentWorkerManager(self.isaac_sim_path)
        self.worker_manager.start_worker(initial_config)
        
        # Capture initial state
        return self.worker_manager.capture_initial_state(initial_config)
        
    def process_state_request(self, config: Dict[str, Any], state_id: str = None) -> Dict[str, str]:
        """Process a state capture request from frontend"""
        if not self.worker_manager:
            raise RuntimeError("Backend not initialized")
            
        return self.worker_manager.update_state_and_capture(config, state_id)
        
    def initialize_animation_mode(self, max_users: int = 8) -> Dict[str, Any]:
        """Initialize animation mode"""
        if not self.worker_manager:
            raise RuntimeError("Backend not initialized")
        return self.worker_manager.initialize_animation_mode(max_users)
        
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
        
    def capture_user_frame(self, user_id: int) -> Dict[str, Any]:
        """Capture frame for user"""
        if not self.worker_manager:
            raise RuntimeError("Backend not initialized")
        return self.worker_manager.capture_user_frame(user_id)
        
    def get_status(self) -> Dict[str, Any]:
        """Get backend status"""
        if not self.worker_manager:
            return {"status": "not_initialized"}
        return self.worker_manager.get_worker_status()
        
    def shutdown(self):
        """Shutdown the backend"""
        if self.worker_manager:
            self.worker_manager.stop_worker()
            self.worker_manager = None

if __name__ == "__main__":
    # Example usage
    isaac_sim_path = "/path/to/isaac-sim"  # Update this
    
    initial_config = {
        "usd_path": "/path/to/environment.usd",
        "robot_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "object_poses": {
            "Cube_01": {"pos": [0.5, 0.0, 0.1], "rot": [0, 0, 0, 1]},
            "Cube_02": {"pos": [0.5, 0.2, 0.1], "rot": [0, 0, 0, 1]},
            "Tennis": {"pos": [0.5, -0.2, 0.1], "rot": [0, 0, 0, 1]}
        }
    }
    
    # Use context manager for automatic cleanup
    with PersistentWorkerManager(isaac_sim_path) as manager:
        manager.start_worker(initial_config)
        
        # State cycling example
        for i in range(5):
            # Modify config for different states
            config = initial_config.copy()
            config["robot_joints"] = [0.1 * i] * 7
            
            result = manager.update_state_and_capture(config, f"test_state_{i}")
            print(f"State {i}: {result}")