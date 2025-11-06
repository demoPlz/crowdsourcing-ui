"""
Demo Video Management Module

Handles demo video recording, display, and prompt sequence snapshot functionality.
Manages video file operations, indexing, and configuration for the crowd interface.
"""

import os
import re
import mimetypes
from pathlib import Path
from threading import Lock


class DemoVideoManager:
    """
    Manages demo video recording and display functionality.
    
    Responsibilities:
    - Demo video recording configuration and file management
    - Demo video display (read-only mode)
    - Prompt sequence snapshot saving (important-state cam_main images)
    - Video file indexing and lookup
    
    Attributes:
        record_demo_videos: Whether demo video recording is enabled
        show_demo_videos: Whether demo video display is enabled
        save_maincam_sequence: Whether important-state snapshots should be saved
        task_name: Task name for directory organization
    """
    
    def __init__(
        self,
        task_name: str | None = None,
        record_demo_videos: bool = False,
        demo_videos_dir: str | None = None,
        demo_videos_clear: bool = False,
        show_demo_videos: bool = False,
        show_videos_dir: str | None = None,
        save_maincam_sequence: bool = False,
        prompt_sequence_dir: str | None = None,
        prompt_sequence_clear: bool = False,
        repo_root: Path | None = None
    ):
        """
        Initialize demo video manager.
        
        Args:
            task_name: Task name for directory organization (e.g., "drawer")
            record_demo_videos: Enable demo video recording
            demo_videos_dir: Directory for saving recorded videos (optional)
            demo_videos_clear: Clear demo videos directory on init
            show_demo_videos: Enable demo video display (read-only)
            show_videos_dir: Directory containing videos to display (optional)
            save_maincam_sequence: Enable important-state snapshot saving
            prompt_sequence_dir: Directory for saving snapshots (optional)
            prompt_sequence_clear: Clear prompt sequence directory on init
            repo_root: Repository root path (defaults to parent of this file)
        """
        self.task_name = task_name
        self.repo_root = repo_root or (Path(__file__).resolve().parent / "..").resolve()
        
        # --- Demo video recording ---
        self.record_demo_videos = bool(record_demo_videos)
        self._demo_videos_dir = None
        self._video_index_lock = Lock()
        self._video_index = 1  # 1-based, reset on each process start
        
        if self.record_demo_videos:
            if demo_videos_dir:
                self._demo_videos_dir = Path(demo_videos_dir).resolve()
            else:
                # Default: data/prompts/{task_name}/demos/
                task_dir = self._task_dir()
                self._demo_videos_dir = task_dir / "demos"
            
            try:
                self._demo_videos_dir.mkdir(parents=True, exist_ok=True)
                if demo_videos_clear:
                    import shutil
                    for item in self._demo_videos_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    print(f"🗑️  Cleared demo videos directory: {self._demo_videos_dir}")
                    self._video_index = 1
                else:
                    self._video_index = self._compute_next_video_index()
                print(f"🎥 Demo video recording enabled → {self._demo_videos_dir} (next index {self._video_index})")
            except Exception as e:
                print(f"⚠️ Failed to initialize demo videos directory: {e}")
                self.record_demo_videos = False
        
        # --- Demo video display ---
        self.show_demo_videos = bool(show_demo_videos or int(os.getenv("SHOW_DEMO_VIDEOS", "0")))
        self._show_videos_dir = None
        self._show_video_exts = (".webm",)  # VP9-only
        
        if self.show_demo_videos:
            if show_videos_dir:
                self._show_videos_dir = Path(show_videos_dir).resolve()
            else:
                task_dir = self._task_dir()
                self._show_videos_dir = task_dir / "demos"
            
            try:
                self._show_videos_dir.mkdir(parents=True, exist_ok=True)
                print(f"📺 Demo video display enabled ← {self._show_videos_dir}")
            except Exception as e:
                print(f"⚠️ Failed to initialize show videos directory: {e}")
                self.show_demo_videos = False
        
        # --- Important-state cam_main image sequence sink ---
        self.save_maincam_sequence = bool(save_maincam_sequence)
        self._prompt_seq_dir = Path(prompt_sequence_dir or "data/prompts/drawer/snapshots").resolve()
        self._prompt_seq_index = 1
        # Track which states have been saved to maintain chronological ordering
        self._saved_sequence_states: set[tuple[str, int]] = set()  # (episode_id, state_id)
        self._max_saved_state_id: int | None = None
        
        if self.save_maincam_sequence:
            try:
                self._prompt_seq_dir.mkdir(parents=True, exist_ok=True)
                if prompt_sequence_clear:
                    import shutil
                    for item in self._prompt_seq_dir.iterdir():
                        if item.is_file() and item.suffix.lower() in (".jpg", ".jpeg", ".png"):
                            item.unlink()
                    print(f"🗑️  Cleared prompt sequence directory: {self._prompt_seq_dir}")
                    self._prompt_seq_index = 1
                else:
                    self._prompt_seq_index = self._compute_next_prompt_seq_index()
                print(f"📸 Important-state capture → {self._prompt_seq_dir} (next index {self._prompt_seq_index:06d})")
            except Exception as e:
                print(f"⚠️ Failed to initialize prompt sequence directory: {e}")
                self.save_maincam_sequence = False
    
    def _prompts_root_dir(self) -> Path:
        """Root folder containing prompts/."""
        return (self.repo_root / "data" / "prompts").resolve()
    
    def _task_dir(self) -> Path:
        """Task-specific directory (prompts/{task_name}/)."""
        tn = self.task_name or "default"
        return (self._prompts_root_dir() / tn).resolve()
    
    def _compute_next_prompt_seq_index(self) -> int:
        """
        Scan the target directory and return next numeric index (1-based).
        Accepts files like 000001.jpg / 42.png / 7.jpeg, ignoring non-numeric stems.
        """
        nums = []
        for p in self._prompt_seq_dir.iterdir():
            if not p.is_file():
                continue
            m = re.match(r"^(\d+)$", p.stem)
            if m:
                nums.append(int(m.group(1)))
        return (max(nums) + 1) if nums else 1
    
    def _compute_next_video_index(self) -> int:
        """
        Scan current videos dir and return the next integer index.
        Accepts files named like '1.webm', '2.mp4', etc.
        If directory is empty (typical after clear), returns 1.
        """
        if not self._demo_videos_dir:
            return 1
        max_idx = 0
        try:
            for p in self._demo_videos_dir.iterdir():
                if not p.is_file():
                    continue
                m = re.match(r"^(\d+)", p.stem)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        except Exception:
            pass
        return (max_idx + 1) if max_idx > 0 else 1
    
    def next_video_filename(self, ext: str) -> tuple[str, int]:
        """
        Return ('{index}{ext}', index) and atomically increment the counter.
        
        Args:
            ext: File extension (e.g., ".webm" or "webm")
        
        Returns:
            Tuple of (filename, index) where filename is like "1.webm"
        """
        if not ext.startswith("."):
            ext = "." + ext
        with self._video_index_lock:
            idx = self._video_index
            self._video_index += 1
        return f"{idx}{ext}", idx
    
    def find_show_video_by_id(self, video_id: int | str) -> tuple[Path | None, str | None]:
        """
        VP9-only: resolve <id>.webm inside the show_videos_dir and return its path + mime.
        
        Args:
            video_id: Video ID (numeric)
        
        Returns:
            Tuple of (path, mime_type) or (None, None) if not found
        """
        vid = str(video_id).strip()
        if not vid.isdigit() or not self._show_videos_dir:
            return None, None
        
        p = self._show_videos_dir / f"{vid}.webm"
        if not p.is_file():
            return None, None
        
        mime = mimetypes.guess_type(str(p))[0] or "video/webm"
        return p, mime
    
    def find_latest_show_video(self) -> tuple[Path | None, str | None]:
        """
        Return (path, id_str) of the latest .webm in _show_videos_dir.
        Files must be named like '<number>.webm' (e.g., 1.webm, 2.webm).
        
        Returns:
            Tuple of (path, id_str) or (None, None) if not found
        """
        try:
            d = self._show_videos_dir
            if not d:
                return None, None
            latest_path = None
            latest_id = None
            for p in d.iterdir():
                if not p.is_file() or p.suffix.lower() != ".webm":
                    continue
                m = re.match(r"^(\d+)$", p.stem)
                if m:
                    num = int(m.group(1))
                    if latest_id is None or num > int(latest_id):
                        latest_id = str(num)
                        latest_path = p
            return latest_path, latest_id
        except Exception:
            return None, None
    
    def get_demo_video_config(self) -> dict:
        """
        Small, stable contract the frontend can consume.
        VP9-only: prefer .webm (VP9) and only accept VP9/WebM uploads.
        
        Returns:
            Configuration dict with keys:
            - enabled: bool
            - task_name: str | None
            - save_dir_abs: str | None
            - save_dir_rel: str | None
            - upload_url: str | None
            - preferred_extension: str
            - preferred_mime: str
            - suggest_canvas_capture: bool
            - filename_pattern: str
            - sequence_start_index: int
            - reset_numbering_each_run: bool
            - accept_mimes: list[str]
        """
        cfg = {
            "enabled": bool(self.record_demo_videos),
            "task_name": self.task_name,
            "save_dir_abs": None,
            "save_dir_rel": None,
            "upload_url": "/api/upload-demo-video" if self.record_demo_videos else None,
            "preferred_extension": "webm",
            "preferred_mime": "video/webm",
            "suggest_canvas_capture": True,
            "filename_pattern": "{index}.{ext}",
            "sequence_start_index": 1,
            "reset_numbering_each_run": True,
            "accept_mimes": ["video/webm"]  # VP9-only
        }
        if self.record_demo_videos and self._demo_videos_dir:
            cfg["save_dir_abs"] = str(self._demo_videos_dir)
            cfg["save_dir_rel"] = self._rel_path_from_repo(self._demo_videos_dir)
        return cfg
    
    def _rel_path_from_repo(self, p: str | Path | None) -> str | None:
        """
        Convert absolute path to repo-relative path.
        
        Args:
            p: Absolute path
        
        Returns:
            Relative path from repo root, or basename if outside repo
        """
        if not p:
            return None
        try:
            rp = Path(p).resolve()
            return str(rp.relative_to(self.repo_root))
        except Exception:
            # If not inside the repo root, return the basename as a safe hint.
            return os.path.basename(str(p))
