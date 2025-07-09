# core/web_file_manager.py
import streamlit as st
import tempfile
import os
import zipfile
from io import BytesIO
from typing import Dict, List, Optional
import logging

class WebFileManager:
    """Web-based file manager for Streamlit deployment"""
    
    def __init__(self):
        # Use session state to store file data
        if "file_cache" not in st.session_state:
            st.session_state.file_cache = {}
        if "output_files" not in st.session_state:
            st.session_state.output_files = {}
    
    def save_genbank_content(self, accession: str, content: str) -> bool:
        """Save GenBank content to session state"""
        try:
            st.session_state.file_cache[f"{accession}.gb"] = {
                "content": content,
                "type": "genbank",
                "size": len(content.encode('utf-8'))
            }
            logging.info(f"Cached GenBank file: {accession}.gb")
            return True
        except Exception as e:
            logging.error(f"Error caching GenBank file: {e}")
            return False
    
    def get_genbank_content(self, accession: str) -> Optional[str]:
        """Get GenBank content from session state"""
        filename = f"{accession}.gb"
        if filename in st.session_state.file_cache:
            return st.session_state.file_cache[filename]["content"]
        return None
    
    def genbank_exists(self, accession: str) -> bool:
        """Check if GenBank file exists in cache"""
        filename = f"{accession}.gb"
        return filename in st.session_state.file_cache
    
    def save_csv_content(self, accession: str, csv_content: str) -> bool:
        """Save CSV content to session state"""
        try:
            filename = f"{accession}_genes.csv"
            st.session_state.output_files[filename] = {
                "content": csv_content,
                "type": "csv",
                "size": len(csv_content.encode('utf-8'))
            }
            logging.info(f"Cached output file: {filename}")
            return True
        except Exception as e:
            logging.error(f"Error caching CSV file: {e}")
            return False
    
    def get_csv_content(self, accession: str) -> Optional[str]:
        """Get CSV content from session state"""
        filename = f"{accession}_genes.csv"
        if filename in st.session_state.output_files:
            return st.session_state.output_files[filename]["content"]
        return None
    
    def list_cached_files(self) -> Dict[str, List[Dict]]:
        """List all cached files"""
        return {
            "genbank_files": [
                {
                    "name": name,
                    "size": info["size"],
                    "type": info["type"]
                }
                for name, info in st.session_state.file_cache.items()
            ],
            "output_files": [
                {
                    "name": name,
                    "size": info["size"],
                    "type": info["type"]
                }
                for name, info in st.session_state.output_files.items()
            ]
        }
    
    def create_download_package(self) -> BytesIO:
        """Create a ZIP package with all files for download"""
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add GenBank files
            for filename, file_info in st.session_state.file_cache.items():
                zip_file.writestr(f"genbank_files/{filename}", file_info["content"])
            
            # Add output files
            for filename, file_info in st.session_state.output_files.items():
                zip_file.writestr(f"output/{filename}", file_info["content"])
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def clear_cache(self):
        """Clear all cached files"""
        st.session_state.file_cache = {}
        st.session_state.output_files = {}
        logging.info("Cleared file cache")
    
    def get_file_info(self, filename: str, file_type: str = "output") -> Dict:
        """Get file information"""
        cache = st.session_state.output_files if file_type == "output" else st.session_state.file_cache
        
        if filename in cache:
            return {
                "exists": True,
                "size": cache[filename]["size"],
                "type": cache[filename]["type"]
            }
        return {"exists": False}
    
    def upload_genbank_file(self, uploaded_file) -> Optional[str]:
        """Handle uploaded GenBank file"""
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode('utf-8')
                filename = uploaded_file.name
                
                # Extract accession from filename or content
                accession = filename.replace('.gb', '').replace('.genbank', '')
                
                st.session_state.file_cache[filename] = {
                    "content": content,
                    "type": "genbank",
                    "size": len(content.encode('utf-8'))
                }
                
                return accession
            except Exception as e:
                logging.error(f"Error uploading file: {e}")
                return None
        return None