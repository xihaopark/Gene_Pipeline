# core/file_manager.py
import os
import logging
from pathlib import Path

class FileManager:
    """Manage local files for GenBank processing"""
    
    def __init__(self, base_dir: str = "genbank_data"):
        self.base_dir = Path(base_dir)
        self.genbank_dir = self.base_dir / "genbank_files"
        self.output_dir = self.base_dir / "output"
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        self.base_dir.mkdir(exist_ok=True)
        self.genbank_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        logging.info(f"Directories setup: {self.base_dir}")
    
    def save_genbank_file(self, accession: str, content: str) -> str:
        """Save GenBank content to local file"""
        filename = f"{accession}.gb"
        filepath = self.genbank_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"Saved GenBank file: {filepath}")
            return str(filepath)
        except Exception as e:
            logging.error(f"Error saving GenBank file: {e}")
            return None
    
    def get_genbank_file_path(self, accession: str) -> str:
        """Get path to GenBank file"""
        filename = f"{accession}.gb"
        return str(self.genbank_dir / filename)
    
    def genbank_file_exists(self, accession: str) -> bool:
        """Check if GenBank file exists locally"""
        filepath = self.genbank_dir / f"{accession}.gb"
        return filepath.exists()
    
    def get_output_file_path(self, accession: str, suffix: str = "genes.csv") -> str:
        """Get path for output file"""
        filename = f"{accession}_{suffix}"
        return str(self.output_dir / filename)
    
    def list_genbank_files(self) -> list:
        """List all GenBank files"""
        return [f.name for f in self.genbank_dir.glob("*.gb")]
    
    def list_output_files(self) -> list:
        """List all output files"""
        return [f.name for f in self.output_dir.glob("*.csv")]
    
    def get_file_info(self, filepath: str) -> dict:
        """Get file information"""
        path = Path(filepath)
        if path.exists():
            stat = path.stat()
            return {
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "exists": True
            }
        return {"exists": False}