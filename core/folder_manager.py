# core/folder_manager.py
import os
import streamlit as st
from pathlib import Path
import platform

class FolderManager:
    """Manages download folders and file storage for GenBank processing"""
    
    def __init__(self):
        self.default_download_folder = self._get_default_download_folder()
        
    def _get_default_download_folder(self) -> str:
        """Get system default download folder"""
        system = platform.system()
        
        if system == "Windows":
            download_path = Path.home() / "Downloads"
        elif system == "Darwin":  # macOS
            download_path = Path.home() / "Downloads"
        else:  # Linux
            download_path = Path.home() / "Downloads"
        
        # If Downloads folder doesn't exist, use home directory
        if not download_path.exists():
            download_path = Path.home()
            
        return str(download_path)
    
    def get_download_folder(self) -> str:
        """Get currently configured download folder"""
        if "download_folder" not in st.session_state:
            st.session_state.download_folder = self.default_download_folder
        return st.session_state.download_folder
    
    def set_download_folder(self, folder_path: str) -> bool:
        """Set download folder path"""
        try:
            folder_path = Path(folder_path)
            if folder_path.exists() and folder_path.is_dir():
                st.session_state.download_folder = str(folder_path)
                return True
            else:
                return False
        except Exception:
            return False
    
    def create_project_folder(self, project_name: str) -> str:
        """Create project folder structure in download directory"""
        base_folder = self.get_download_folder()
        project_folder = Path(base_folder) / f"GenBank_Analysis_{project_name}"
        
        # Create folder structure
        project_folder.mkdir(exist_ok=True)
        (project_folder / "genbank_files").mkdir(exist_ok=True)
        (project_folder / "output").mkdir(exist_ok=True)
        (project_folder / "logs").mkdir(exist_ok=True)
        
        return str(project_folder)
    
    def get_project_folder(self) -> str:
        """Get current project folder"""
        project_name = st.session_state.get("project_name", "default_project")
        return self.create_project_folder(project_name)
    
    def list_project_files(self, project_name: str = None) -> dict:
        """List files in project folder"""
        if not project_name:
            project_name = st.session_state.get("project_name", "default_project")
        
        project_folder = Path(self.get_download_folder()) / f"GenBank_Analysis_{project_name}"
        
        if not project_folder.exists():
            return {"genbank_files": [], "output_files": [], "log_files": []}
        
        return {
            "genbank_files": list((project_folder / "genbank_files").glob("*.gb")),
            "output_files": list((project_folder / "output").glob("*.csv")),
            "log_files": list((project_folder / "logs").glob("*.log"))
        }
    
    def render_folder_selection_ui(self):
        """Render folder selection UI components"""
        st.subheader("📁 File Storage Settings")
        
        current_folder = self.get_download_folder()
        st.write(f"**Current Download Folder:** `{current_folder}`")
        
        # Folder selection interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            new_folder = st.text_input(
                "Custom Download Folder Path:",
                value=current_folder,
                help="Enter full folder path or use default download folder"
            )
        
        with col2:
            if st.button("📂 Set Folder", help="Set new download folder"):
                if self.set_download_folder(new_folder):
                    st.success("✅ Folder set successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid folder path!")
        
        # Reset to default
        if st.button("🔄 Reset to Default Download Folder"):
            st.session_state.download_folder = self.default_download_folder
            st.success("✅ Reset to default download folder")
            st.rerun()
        
        # Project folder configuration
        st.write("---")
        project_name = st.text_input(
            "Project Name:",
            value=st.session_state.get("project_name", ""),
            placeholder="Enter project name to create dedicated folder"
        )
        
        if project_name:
            st.session_state.project_name = project_name
            project_folder = self.create_project_folder(project_name)
            st.info(f"📁 Project Folder: `{project_folder}`")
            
            # Display folder structure
            with st.expander("📋 Folder Structure"):
                st.code(f"""
{project_folder}/
├── genbank_files/    # GenBank file storage
├── output/          # Analysis results output
└── logs/           # Processing logs
                """)
            
            # List existing files
            files = self.list_project_files(project_name)
            if any(files.values()):
                with st.expander("📄 Existing Files"):
                    st.write(f"**GenBank Files:** {len(files['genbank_files'])}")
                    st.write(f"**Output Files:** {len(files['output_files'])}")
                    st.write(f"**Log Files:** {len(files['log_files'])}")
    
    def get_genbank_folder(self) -> Path:
        """Get GenBank files folder for current project"""
        project_folder = Path(self.get_project_folder())
        return project_folder / "genbank_files"
    
    def get_output_folder(self) -> Path:
        """Get output folder for current project"""
        project_folder = Path(self.get_project_folder())
        return project_folder / "output"
    
    def get_logs_folder(self) -> Path:
        """Get logs folder for current project"""
        project_folder = Path(self.get_project_folder())
        return project_folder / "logs"