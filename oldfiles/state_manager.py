# core/state_manager.py
import streamlit as st

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        "accession": "",
        "assemblies": [],
        "selected_assembly": None,
        "genes": [],
        "genbank_accession": ""
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_all():
    """Reset all session state"""
    keys_to_clear = [
        "accession", "assemblies", "selected_assembly", 
        "genes", "genbank_accession"
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]