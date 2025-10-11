import os
import base64
import torch
import shutil
import tempfile
from pathlib import Path

class Database():
    def __init__(self, cache_root: str | None = None):
        """
        Initialize database for persisting observations and camera views to disk.
        
        Args:
            cache_root: Optional custom cache directory. Defaults to temp/crowd_obs_cache
        """
        # Set CROWD_OBS_CACHE to override where temporary per-state observations are stored.
        if cache_root:
            self._obs_cache_root = Path(cache_root).resolve()
        else:
            default_cache = os.path.join(tempfile.gettempdir(), "crowd_obs_cache")
            self._obs_cache_root = Path(os.getenv("CROWD_OBS_CACHE", default_cache))
        
        try:
            self._obs_cache_root.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Failed to create cache directory {self._obs_cache_root}: {e}")

    def _episode_cache_dir(self, episode_id: str) -> Path:
        """Get or create cache directory for a specific episode."""
        d = self._obs_cache_root / str(episode_id)
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return d

    def persist_views_to_disk(self, episode_id: str, state_id: int, views_b64: dict[str, str]) -> dict[str, str]:
        """
        Persist base64 (data URL) JPEGs for each camera to disk.
        
        Args:
            episode_id: Episode identifier
            state_id: State identifier within episode
            views_b64: Dict mapping camera_name -> base64 data URL
            
        Returns:
            Dict mapping camera_name -> absolute file path
        """
        if not views_b64:
            return {}
        
        out: dict[str, str] = {}
        try:
            d = self._episode_cache_dir(episode_id) / "views"
            d.mkdir(parents=True, exist_ok=True)
            
            for cam, data_url in views_b64.items():
                # Expect "data:image/jpeg;base64,....."
                if not isinstance(data_url, str):
                    continue
                idx = data_url.find("base64,")
                if idx == -1:
                    continue
                b64 = data_url[idx + len("base64,"):]
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    continue
                    
                path = d / f"{state_id}_{cam}.jpg"
                with open(path, "wb") as f:
                    f.write(raw)
                out[cam] = str(path)
                
        except Exception as e:
            print(f"⚠️ Failed to persist views ep={episode_id} state={state_id}: {e}")
        return out

    def load_views_from_disk(self, view_paths: dict[str, str]) -> dict[str, str]:
        """
        Load per-camera JPEG files and return data URLs.
        
        Args:
            view_paths: Dict mapping camera_name -> file path
            
        Returns:
            Dict mapping camera_name -> base64 data URL
        """
        if not view_paths:
            return {}
            
        out: dict[str, str] = {}
        for cam, path in view_paths.items():
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                out[cam] = f"data:image/jpeg;base64,{b64}"
            except Exception:
                # Missing/removed file → skip this camera
                pass
        return out

    def persist_obs_to_disk(self, episode_id: str, state_id: int, obs: dict) -> str | None:
        """
        Writes the observations dict to a single file for the state and returns the path.
        
        Args:
            episode_id: Episode identifier
            state_id: State identifier within episode
            obs: Observations dictionary (may contain tensors, numpy arrays, etc.)
            
        Returns:
            Absolute path to saved file, or None if failed
        """
        try:
            p = self._episode_cache_dir(episode_id) / f"{state_id}.pt"
            # Tensors/ndarrays/py objects handled by torch.save
            torch.save(obs, p)
            return str(p)
        except Exception as e:
            print(f"⚠️ Failed to persist obs ep={episode_id} state={state_id}: {e}")
            return None

    def load_obs_from_disk(self, path: str | None) -> dict:
        """
        Load observations from disk file.
        
        Args:
            path: Path to observations file
            
        Returns:
            Observations dictionary, or empty dict if failed
        """
        if not path:
            return {}
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"⚠️ Failed to load obs from {path}: {e}")
            return {}

    def delete_obs_from_disk(self, path: str | None):
        """
        Delete an observations file from disk.
        
        Args:
            path: Path to observations file to delete
        """
        if not path:
            return
        try:
            os.remove(path)
        except Exception:
            pass

    def purge_episode_cache(self, episode_id: str):
        """
        Remove the entire temp cache folder for an episode.
        
        Args:
            episode_id: Episode identifier to purge
        """
        try:
            d = self._episode_cache_dir(episode_id)
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
                print(f"🗑️ Purged cache for episode {episode_id}")
        except Exception as e:
            print(f"⚠️ Failed to purge episode cache {episode_id}: {e}")

    def get_cache_stats(self) -> dict:
        """
        Get statistics about the cache directory.
        
        Returns:
            Dict with cache statistics
        """
        stats = {
            "cache_root": str(self._obs_cache_root),
            "total_episodes": 0,
            "total_size_mb": 0.0,
            "episodes": {}
        }
        
        try:
            if not self._obs_cache_root.exists():
                return stats
                
            total_size = 0
            for episode_dir in self._obs_cache_root.iterdir():
                if episode_dir.is_dir():
                    episode_id = episode_dir.name
                    episode_size = 0
                    file_count = 0
                    
                    for file_path in episode_dir.rglob("*"):
                        if file_path.is_file():
                            file_size = file_path.stat().st_size
                            episode_size += file_size
                            file_count += 1
                    
                    stats["episodes"][episode_id] = {
                        "size_mb": episode_size / (1024 * 1024),
                        "file_count": file_count
                    }
                    total_size += episode_size
                    stats["total_episodes"] += 1
            
            stats["total_size_mb"] = total_size / (1024 * 1024)
            
        except Exception as e:
            print(f"⚠️ Error calculating cache stats: {e}")
            
        return stats

    def cleanup_old_episodes(self, keep_latest: int = 10):
        """
        Clean up old episode caches, keeping only the most recent ones.
        
        Args:
            keep_latest: Number of latest episodes to keep
        """
        try:
            if not self._obs_cache_root.exists():
                return
                
            # Get all episode directories sorted by modification time (newest first)
            episode_dirs = []
            for episode_dir in self._obs_cache_root.iterdir():
                if episode_dir.is_dir():
                    episode_dirs.append((episode_dir.stat().st_mtime, episode_dir))
            
            episode_dirs.sort(reverse=True)  # Newest first
            
            # Remove old episodes beyond keep_latest
            removed_count = 0
            for i, (_, episode_dir) in enumerate(episode_dirs):
                if i >= keep_latest:
                    try:
                        shutil.rmtree(episode_dir, ignore_errors=True)
                        removed_count += 1
                        print(f"🗑️ Cleaned up old episode cache: {episode_dir.name}")
                    except Exception as e:
                        print(f"⚠️ Failed to remove {episode_dir}: {e}")
            
            if removed_count > 0:
                print(f"🧹 Cleaned up {removed_count} old episode caches")
                
        except Exception as e:
            print(f"⚠️ Error during cache cleanup: {e}")