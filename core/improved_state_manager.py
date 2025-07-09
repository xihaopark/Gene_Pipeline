# core/improved_state_manager.py
import streamlit as st

class ImprovedStateManager:
    """Enhanced state management with auto-initialization"""
    
    @staticmethod
    def init_session_state():
        """Automatically initialize all necessary session states"""
        defaults = {
            # Basic search states
            "accession": "",
            "assemblies": [],
            "selected_assembly": None,
            "genes": [],
            "genbank_accession": "",
            
            # KEGG batch processing states
            "kegg_organism": "",
            "kegg_genomes": [],
            "batch_results": None,
            "batch_include_sequences": False,
            
            # File management states
            "download_folder": "",
            "project_name": "",
            
            # Processing stage states
            "processing_stage": "ready",  # ready, downloading, parsing, completed, error
            "downloaded_files": [],
            "parsing_progress": 0,
            
            # Data preview states
            "genbank_summary": {},
            "data_preview": None,
            "download_results": None,
            "parsing_results": None,
            
            # UI states
            "ui_initialized": True,
            "optimization_settings": {},
            
            # File upload states
            "uploaded_accession": "",
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def reset_processing_state():
        """Reset processing-related states while preserving user inputs"""
        processing_keys = [
            "assemblies", "selected_assembly", "genes", "genbank_accession",
            "kegg_genomes", "batch_results", "downloaded_files", 
            "parsing_progress", "genbank_summary", "data_preview",
            "download_results", "parsing_results", "uploaded_accession"
        ]
        
        for key in processing_keys:
            if key in st.session_state:
                if key.endswith(('s', 'files')) and key not in ['parsing_progress', 'batch_include_sequences']:
                    st.session_state[key] = []
                elif key in ['batch_include_sequences']:
                    st.session_state[key] = False
                elif key in ['parsing_progress']:
                    st.session_state[key] = 0
                elif isinstance(st.session_state.get(key), str):
                    st.session_state[key] = ""
                else:
                    st.session_state[key] = None
        
        st.session_state.processing_stage = "ready"
    
    @staticmethod
    def is_ready_for_search() -> bool:
        """Check if system is ready for search operations"""
        return st.session_state.get("ui_initialized", False)
    
    @staticmethod
    def update_processing_stage(stage: str, data: dict = None):
        """Update current processing stage with optional data"""
        valid_stages = ["ready", "downloading", "parsing", "completed", "error"]
        if stage in valid_stages:
            st.session_state.processing_stage = stage
        
        if data:
            for key, value in data.items():
                st.session_state[key] = value
    
    @staticmethod
    def get_processing_stage() -> str:
        """Get current processing stage"""
        return st.session_state.get("processing_stage", "ready")
    
    @staticmethod
    def render_processing_status():
        """Render processing status indicator in sidebar"""
        stage = st.session_state.get("processing_stage", "ready")
        
        status_icons = {
            "ready": "🟢",
            "downloading": "🔄",
            "parsing": "⚙️",
            "completed": "✅",
            "error": "❌"
        }
        
        status_texts = {
            "ready": "Ready",
            "downloading": "Downloading...",
            "parsing": "Parsing...",
            "completed": "Completed",
            "error": "Error"
        }
        
        icon = status_icons.get(stage, "🔘")
        text = status_texts.get(stage, "Unknown")
        
        st.sidebar.write(f"**Status:** {icon} {text}")
        
        # Display progress information
        if stage == "downloading" and st.session_state.get("downloaded_files"):
            downloaded = len(st.session_state.downloaded_files)
            st.sidebar.write(f"Downloaded: {downloaded} files")
        
        elif stage == "parsing":
            progress = st.session_state.get("parsing_progress", 0)
            if progress > 0:
                st.sidebar.progress(progress / 100)
                st.sidebar.write(f"Progress: {progress}%")
    
    @staticmethod
    def save_state(key: str, value):
        """Save a value to session state"""
        st.session_state[key] = value
    
    @staticmethod
    def get_state(key: str, default=None):
        """Get a value from session state with default"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def has_state(key: str) -> bool:
        """Check if a key exists in session state"""
        return key in st.session_state
    
    @staticmethod
    def clear_state(key: str):
        """Clear a specific state key"""
        if key in st.session_state:
            del st.session_state[key]