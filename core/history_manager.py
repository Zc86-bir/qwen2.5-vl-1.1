from datetime import datetime
import pickle
import csv
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

class HistoryManager:
    """Manages conversation and analysis history"""
    def __init__(self, save_dir="history"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.current_session = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_entry(self, entry):
        """Add a new entry to the current session"""
        self.current_session.append({
            "timestamp": datetime.now().isoformat(),
            "content": entry
        })
        return len(self.current_session) - 1  # Return index of added entry

    def get_entry(self, index):
        """Get entry by index"""
        if 0 <= index < len(self.current_session):
            return self.current_session[index]
        return None

    def save_session(self, session_id=None):
        """Save current session to file"""
        if session_id:
            self.session_id = session_id
            
        file_path = self.save_dir / f"session_{self.session_id}.pkl"
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self.current_session, f)
            logger.info(f"Session saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
        
    def load_session(self, session_id):
        """Load session from file"""
        file_path = self.save_dir / f"session_{session_id}.pkl"
        if file_path.exists():
            try:
                with open(file_path, "rb") as f:
                    self.current_session = pickle.load(f)
                self.session_id = session_id
                logger.info(f"Session loaded from {file_path}")
                return self.current_session
            except Exception as e:
                logger.error(f"Error loading session: {e}")
                return []
        else:
            logger.warning(f"Session file not found: {file_path}")
            return []

    def list_sessions(self):
        """List all available session files"""
        sessions = []
        for file_path in self.save_dir.glob("session_*.pkl"):
            session_id = file_path.stem.replace("session_", "")
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                    
                # Get first and last timestamp
                first_timestamp = data[0]["timestamp"] if data else None
                last_timestamp = data[-1]["timestamp"] if data else None
                entry_count = len(data)
                
                sessions.append({
                    "id": session_id,
                    "first_timestamp": first_timestamp,
                    "last_timestamp": last_timestamp,
                    "entries": entry_count,
                    "file_path": str(file_path)
                })
            except Exception as e:
                logger.error(f"Error reading session {session_id}: {e}")
                
        return sorted(sessions, key=lambda x: x["id"], reverse=True)

    def export_csv(self, file_path):
        """Export current session to CSV file"""
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Content Type", "Content"])
                for entry in self.current_session:
                    content = entry["content"]
                    if isinstance(content, dict):
                        content_type = "dict"
                        content_str = json.dumps(content, ensure_ascii=False)
                    else:
                        content_type = type(content).__name__
                        content_str = str(content)
                    writer.writerow([entry["timestamp"], content_type, content_str])
            logger.info(f"Session exported to CSV: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return False

    def export_json(self, file_path):
        """Export current session to JSON file"""
        try:
            # Convert session data to serializable format
            serializable_session = []
            for entry in self.current_session:
                serialized_entry = {
                    "timestamp": entry["timestamp"],
                    "content_type": type(entry["content"]).__name__
                }
                
                # Handle different content types
                if isinstance(entry["content"], dict):
                    serialized_entry["content"] = entry["content"]
                elif isinstance(entry["content"], list):
                    serialized_entry["content"] = entry["content"]
                else:
                    serialized_entry["content"] = str(entry["content"])
                    
                serializable_session.append(serialized_entry)
            
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(serializable_session, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Session exported to JSON: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            return False

    def clear(self):
        """Clear current session"""
        self.current_session = []
        return True