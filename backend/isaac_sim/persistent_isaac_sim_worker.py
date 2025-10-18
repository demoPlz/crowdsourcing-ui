#!/usr/bin/env python3
"""
Persistent Isaac Sim worker that stays running and processes commands via file communication
This script runs in Isaac Sim's Python environment and communicates with backend via JSON files
"""

import sys
import json
import argparse
import time
import os
import signal
from pathlib import Path
from isaacsim import SimulationApp

# Import our worker class (now safe since we're in Isaac environment)
from isaac_sim_worker import IsaacSimWorker

class PersistentWorker:
    """Long-running worker that processes commands from backend"""
    
    def __init__(self, output_dir: str, max_users: int = 8, simulation_app=None):
        self.output_dir = output_dir
        self.max_users = max_users
        self.isaac_worker = IsaacSimWorker(simulation_app=simulation_app)
        # Set communication directory so Isaac worker can check for direct commands
        self.isaac_worker.worker_communication_dir = output_dir
        self.running = True
        
        # Communication files
        self.command_file = f"{output_dir}/commands.json"
        self.result_file = f"{output_dir}/results.json"
        self.status_file = f"{output_dir}/status.json"
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
    def start(self, initial_config: dict):
        """Start the persistent worker loop"""
        print("WORKER_READY")  # Signal to manager that we're ready
        sys.stdout.flush()
        
        # Write initial status
        self._update_status("ready", "Worker started and ready for commands")
        
        # Main command processing loop
        while self.running:
            try:
                # PRIORITY 1: Always check for new commands first
                command = self._check_for_command()
                if command:
                    result = self._process_command(command)
                    self._send_result(result)
                    # Immediately loop back to check for more commands
                    continue
                
                # PRIORITY 2: Handle animations and chunked generation when no commands are pending
                work_done = False
                
                # Process chunked frame generation (non-blocking)
                if self.isaac_worker.chunked_generation_state:
                    generation_work = self.isaac_worker.process_chunked_frame_generation(frames_per_chunk=3)
                    work_done = work_done or generation_work
                
                # Handle active animations (replay mode)
                if self.isaac_worker.animation_mode and self.isaac_worker.active_animations:
                    # Update physics
                    if self.isaac_worker.world:
                        self.isaac_worker.world.step(render=True)
                    
                    # Update all user animations (joint interpolation)
                    self.isaac_worker.update_animations()
                    work_done = True
                
                if work_done:
                    # Very short delay for responsive command checking when doing work
                    time.sleep(0.005)  # 5ms - allows ~200 Hz command checking
                else:
                    # Moderate delay when idle
                    time.sleep(0.05)  # 50ms - still responsive
                
            except Exception as e:
                print(f"Error in worker loop: {e}")
                import traceback
                traceback.print_exc()
                self._send_result({"status": "error", "message": str(e)})
                
        print("Worker shutting down...")
        self._update_status("shutdown", "Worker shut down gracefully")
        
    def _check_for_command(self) -> dict:
        """Check if there's a new command waiting"""
        command_signal_file = f"{self.command_file}.signal"
        
        if os.path.exists(command_signal_file):
            try:
                # Read command
                with open(self.command_file, 'r') as f:
                    command = json.load(f)
                    
                # Remove signal file to acknowledge
                os.remove(command_signal_file)
                
                print(f"Received command: {command.get('action', 'unknown')}")
                return command
                
            except Exception as e:
                print(f"Error reading command: {e}")
                return None
                
        return None
        
    def _process_command(self, command: dict) -> dict:
        """Process a command and return result"""
        action = command.get('action')
        
        try:
            if action == 'initialize_and_capture':
                # Initialize simulation and capture first state
                config = command['config']
                output_dir = command['output_dir']
                
                print(f"Debug: Before capture_static_images - simulation_initialized = {self.isaac_worker.simulation_initialized}")
                result = self.isaac_worker.capture_static_images(config, output_dir)
                print(f"Debug: After capture_static_images - simulation_initialized = {self.isaac_worker.simulation_initialized}")
                print("SIMULATION_INITIALIZED")  # Signal initialization complete
                sys.stdout.flush()
                
                return {
                    "status": "success",
                    "action": action,
                    "result": result
                }
                
            elif action == 'update_and_capture':
                # Update state and capture images (fast path)
                config = command['config']
                output_dir = command['output_dir']
                
                if not self.isaac_worker.simulation_initialized:
                    # Fallback to full initialization if needed
                    result = self.isaac_worker.capture_static_images(config, output_dir)
                else:
                    # Fast state update
                    result = self.isaac_worker.update_and_capture(config, output_dir)
                    
                return {
                    "status": "success", 
                    "action": action,
                    "result": result
                }
                
            elif action == 'get_status':
                # Return worker status
                return {
                    "status": "success",
                    "worker_ready": True,
                    "simulation_initialized": self.isaac_worker.simulation_initialized,
                    "animation_mode": self.isaac_worker.animation_mode
                }
                
            elif action == 'initialize_animation':
                # Initialize animation mode with cloned environments
                max_users = command.get('max_users', self.max_users)
                
                print(f"Debug: simulation_initialized = {self.isaac_worker.simulation_initialized}")
                if not self.isaac_worker.simulation_initialized:
                    return {"status": "error", "message": "Must initialize simulation first"}
                    
                print(f"Initializing animation mode with {max_users} users...")
                self.isaac_worker.initialize_animation_mode(max_users)
                
                return {
                    "status": "success",
                    "action": action,
                    "message": f"Animation mode initialized with {max_users} users"
                }
                
            elif action == 'start_user_animation':
                # Start animation for specific user
                user_id = command['user_id']
                goal_joints = command.get('goal_joints')
                duration = command.get('duration', 3.0)
                
                result = self.isaac_worker.start_user_animation(
                    user_id=user_id,
                    goal_joints=goal_joints,
                    duration=duration
                )
                
                return {
                    "status": "success" if "error" not in result else "error",
                    "action": action,
                    "result": result
                }
                
            elif action == 'stop_user_animation':
                # Stop animation for specific user
                user_id = command['user_id']
                result = self.isaac_worker.stop_user_animation(user_id)
                
                return {
                    "status": "success",
                    "action": action,
                    "result": result
                }
                
            elif action == 'capture_user_frame':
                # Capture frame for specific user
                user_id = command['user_id']
                output_dir = command['output_dir']
                
                result = self.isaac_worker.capture_user_frame(user_id, output_dir)
                
                if result is None:
                    return {
                        "status": "error",
                        "action": action,
                        "message": f"User {user_id} environment not found"
                    }
                
                return {
                    "status": "success",
                    "action": action,
                    "result": result
                }
                
            elif action == 'start_animation_loop':
                # Start the main animation loop (blocking)
                output_dir = command['output_dir']
                
                # This will block until animation mode is stopped
                self.isaac_worker.animation_loop(output_dir)
                
                return {
                    "status": "success",
                    "action": action,
                    "message": "Animation loop completed"
                }
                
            elif action == 'sync_animation_environments':
                # Synchronize all animation environments to new state
                config = command['config']
                
                if not self.isaac_worker.animation_mode:
                    return {
                        "status": "error",
                        "action": action,
                        "message": "Animation mode not initialized"
                    }
                
                self.isaac_worker.sync_animation_environments(config)
                
                return {
                    "status": "success",
                    "action": action,
                    "message": "Animation environments synchronized"
                }
                
            elif action == 'shutdown':
                # Graceful shutdown
                self.running = False
                return {"status": "success", "message": "Shutting down"}
                
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}
                
        except Exception as e:
            print(f"Error processing command {action}: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error", 
                "action": action,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            
    def _send_result(self, result: dict):
        """Send result back to manager"""
        try:
            # Write result to file
            with open(self.result_file, 'w') as f:
                json.dump(result, f, indent=2)
                
            # Create signal file to indicate result is ready
            result_signal_file = f"{self.result_file}.signal"
            Path(result_signal_file).touch()
            
            print(f"Sent result: {result.get('status', 'unknown')}")
            
        except Exception as e:
            print(f"Error sending result: {e}")
            
    def _update_status(self, status: str, message: str = ""):
        """Update status file"""
        try:
            status_data = {
                "status": status,
                "message": message,
                "timestamp": time.time(),
                "simulation_initialized": getattr(self.isaac_worker, 'simulation_initialized', False)
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            print(f"Error updating status: {e}")

def main():
    parser = argparse.ArgumentParser(description="Persistent Isaac Sim Worker")
    parser.add_argument('--config', type=str, required=True, help='Initial config file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for communication')
    parser.add_argument('--max-users', type=int, default=8, help='Maximum users for animation mode')
    args = parser.parse_args()
    
    # Load initial configuration
    with open(args.config, 'r') as f:
        initial_config = json.load(f)
    
    # Start Isaac Sim
    global simulation_app
    simulation_app = SimulationApp({"headless": True})
    
    try:
        # Create and start persistent worker
        worker = PersistentWorker(args.output_dir, args.max_users, simulation_app)
        worker.start(initial_config)
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Worker error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'simulation_app' in globals():
            simulation_app.close()
            
    print("Persistent worker terminated")

if __name__ == "__main__":
    main()