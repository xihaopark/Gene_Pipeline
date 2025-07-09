# core/config_manager.py
import os
import streamlit as st
from pathlib import Path
from typing import Optional
import logging

class ConfigManager:
    """Secure configuration and API key management"""
    
    @staticmethod
    def get_api_key(key_name: str) -> Optional[str]:
        """
        Get API key from multiple sources in order of preference:
        1. Session state (temporary)
        2. Environment variables
        3. Streamlit secrets
        4. .env file (local development)
        """
        # Check session state first
        session_key = f"{key_name.lower()}_api_key"
        if session_key in st.session_state and st.session_state[session_key]:
            return st.session_state[session_key]
        
        # Check environment variables
        env_value = os.environ.get(key_name.upper())
        if env_value:
            return env_value
        
        # Check Streamlit secrets
        try:
            if hasattr(st, 'secrets') and key_name.upper() in st.secrets:
                return st.secrets[key_name.upper()]
        except:
            pass
        
        # Check .env file (for local development)
        env_file = Path('.env')
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            if key.strip() == key_name.upper():
                                return value.strip()
            except Exception as e:
                logging.error(f"Error reading .env file: {e}")
        
        return None
    
    @staticmethod
    def set_api_key(key_name: str, value: str):
        """Store API key in session state temporarily"""
        session_key = f"{key_name.lower()}_api_key"
        st.session_state[session_key] = value
    
    @staticmethod
    def get_anthropic_api_key() -> Optional[str]:
        """Get Anthropic API key"""
        return ConfigManager.get_api_key('ANTHROPIC_API_KEY')
    
    @staticmethod
    def get_ncbi_api_key() -> Optional[str]:
        """Get NCBI API key"""
        return ConfigManager.get_api_key('NCBI_API_KEY')
    
    @staticmethod
    def get_download_folder() -> str:
        """Get default download folder"""
        folder = ConfigManager.get_api_key('DEFAULT_DOWNLOAD_FOLDER')
        if folder and Path(folder).exists():
            return folder
        
        # Return system default
        return str(Path.home() / "Downloads")
    
    @staticmethod
    def is_debug_mode() -> bool:
        """Check if debug mode is enabled"""
        debug = ConfigManager.get_api_key('DEBUG_MODE')
        return debug and debug.lower() == 'true'
    
    @staticmethod
    def render_api_key_input():
        """Render API key input in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔑 API Configuration")
        
        # Anthropic API Key
        current_key = ConfigManager.get_anthropic_api_key()
        api_key = st.sidebar.text_input(
            "Claude API Key:",
            value=current_key if current_key else "",
            type="password",
            help="Enter your Anthropic Claude API key"
        )
        
        if api_key and api_key != current_key:
            ConfigManager.set_api_key('ANTHROPIC_API_KEY', api_key)
            st.sidebar.success("✅ API key updated")
        
        # Show API key source
        if current_key:
            if 'anthropic_api_key' in st.session_state:
                st.sidebar.caption("📍 Source: Session (temporary)")
            elif 'ANTHROPIC_API_KEY' in os.environ:
                st.sidebar.caption("📍 Source: Environment variable")
            elif hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                st.sidebar.caption("📍 Source: Streamlit secrets")
            else:
                st.sidebar.caption("📍 Source: Local .env file")
        
        # Instructions
        with st.sidebar.expander("🔒 API Key Security"):
            st.markdown("""
            **Best Practices:**
            - Never commit API keys to git
            - Use environment variables for production
            - Use .env file for local development
            - Use Streamlit secrets for cloud deployment
            
            **Get your API key:**
            - Claude: https://console.anthropic.com/
            - NCBI: https://www.ncbi.nlm.nih.gov/account/
            """)
        
        return api_key