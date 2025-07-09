import time
from io import StringIO
import streamlit as st
import sys
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import zipfile
from io import BytesIO

# Configure logging to be silent
logging.basicConfig(level=logging.ERROR)

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import core modules
try:
    from core.genbank_processor import GenBankProcessor
    from core.improved_state_manager import ImprovedStateManager
    from core.kegg_integration import KEGGIntegration
    from core.staged_processor import StagedGenomeProcessor
    from core.web_file_manager import WebFileManager
    from core.smart_parser_generator import SmartParserGenerator
    from core.feature_selector import SequenceMappingFormatter
    from core.sequence_analyzer import SequenceAnalyzer
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

def main():
    st.set_page_config(
        page_title="GenBank Gene Extractor", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Auto-initialize session state
    ImprovedStateManager.init_session_state()
    
    # Initialize default settings (simplified - always use AI)
    if 'parser_method' not in st.session_state:
        st.session_state.parser_method = "Claude AI (Fast)"
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = ['gene', 'CDS']
    
    st.title("🧬 GenBank Gene Extractor")
    st.markdown("*Extract and analyze genomic data with AI-powered processing*")
    
    # Initialize processors
    processor = GenBankProcessor()
    kegg_integration = KEGGIntegration()
    staged_processor = StagedGenomeProcessor()
    
    # Main unified interface
    render_unified_interface(processor, staged_processor)

def render_unified_interface(processor, staged_processor):
    """Unified interface combining all search methods"""
    
    # Search input section
    st.header("🔍 Search for Genomes")
    
    # Two-column layout for search options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Search by Organism Name")
        organism_name = st.text_input(
            "Organism Name:",
            value=st.session_state.get("organism_name", ""),
            placeholder="e.g., Clostridioides difficile",
            help="Search KEGG database for genomes of this organism"
        )
        
        if organism_name:
            st.session_state.organism_name = organism_name
            
            # Parameters for KEGG search
            max_genomes = st.number_input(
                "Max Genomes:",
                min_value=1,
                max_value=20,
                value=5,
                help="Maximum number of genomes to process"
            )
            
            if st.button("🔍 Search by Organism", type="primary", key="kegg_search"):
                run_kegg_search(staged_processor, organism_name, max_genomes)
    
    with col2:
        st.subheader("Search by Accession Number")
        accession = st.text_input(
            "Genome Accession:",
            value=st.session_state.get("accession", ""),
            placeholder="e.g., GCA_000005825.2",
            help="Search by specific genome accession number"
        )
        
        if accession:
            st.session_state.accession = accession
            
            if st.button("🔍 Search by Accession", type="primary", key="direct_search"):
                run_direct_search(processor, accession)
    
    # Results display section
    if st.session_state.get("search_results"):
        render_search_results(processor, staged_processor)
    
    # Processing results section
    if st.session_state.get("processing_results"):
        render_processing_results()
    
    # Sequence mapping section (integrated)
    if st.session_state.get("final_results"):
        render_integrated_sequence_mapping()

def run_kegg_search(staged_processor, organism_name, max_genomes):
    """Execute KEGG search with simplified workflow"""
    with st.spinner("Searching KEGG database..."):
        try:
            # Stage 1: Download
            results = staged_processor.stage1_download_all_genbank_files(
                organism_name, max_genomes
            )
            
            if results['status'] == 'completed':
                st.session_state.search_results = {
                    'type': 'kegg',
                    'data': results,
                    'organism': organism_name
                }
                st.success(f"✅ Found and downloaded {results['successful_downloads']} genomes")
                st.rerun()
            else:
                st.error("❌ Search failed. Please try again.")
                
        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")

def run_direct_search(processor, accession):
    """Execute direct accession search"""
    with st.spinner("Searching NCBI database..."):
        try:
            assemblies = processor.search_assembly(accession)
            if assemblies:
                st.session_state.search_results = {
                    'type': 'direct',
                    'data': assemblies,
                    'accession': accession
                }
                st.success(f"✅ Found {len(assemblies)} assemblies")
                st.rerun()
            else:
                st.error("❌ No assemblies found for this accession")
                
        except Exception as e:
            st.error(f"❌ Error during search: {str(e)}")

def render_search_results(processor, staged_processor):
    """Display search results and processing options"""
    st.header("📊 Search Results")
    
    results = st.session_state.search_results
    
    if results['type'] == 'kegg':
        # KEGG results
        data = results['data']
        st.info(f"🔬 Found {data['successful_downloads']} genomes for **{results['organism']}**")
        
        # Show summary
        if data['successful_downloads'] > 0:
            summary = staged_processor.stage2_generate_summary_report(data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Genomes", summary['successful_files'])
            with col2:
                st.metric("Total Genes", summary['total_genes'])
            with col3:
                st.metric("Total Size", summary['total_size_mb'])
            
            # Process button
            include_sequences = st.checkbox("Include DNA sequences", value=True)
            
            if st.button("🚀 Process All Genomes", type="primary"):
                process_kegg_results(staged_processor, data, include_sequences)
    
    elif results['type'] == 'direct':
        # Direct search results
        assemblies = results['data']
        st.info(f"🔬 Found {len(assemblies)} assemblies for **{results['accession']}**")
        
        # Assembly selection
        selected_assembly = st.selectbox(
            "Select assembly to process:",
            options=range(len(assemblies)),
            format_func=lambda x: f"{assemblies[x]['assembly_acc']} - {assemblies[x]['organism']}",
            key="assembly_selector"
        )
        
        if selected_assembly is not None:
            assembly = assemblies[selected_assembly]
            
            # Show assembly details
            with st.expander("Assembly Details", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Assembly Name:** {assembly.get('assembly_name', 'N/A')}")
                    st.write(f"**Level:** {assembly.get('assembly_level', 'N/A')}")
                with col2:
                    st.write(f"**Submitter:** {assembly.get('submitter', 'N/A')}")
                    st.write(f"**Version:** {assembly.get('version', 'N/A')}")
            
            if st.button("🚀 Process Selected Assembly", type="primary"):
                process_direct_results(processor, assembly)

def process_kegg_results(staged_processor, data, include_sequences):
    """Process KEGG search results"""
    with st.spinner("Processing genomes with AI..."):
        try:
            # Stage 3: Parse with AI (default)
            results = staged_processor.stage3_batch_parse_files(data, include_sequences)
            
            if results['status'] == 'completed':
                st.session_state.processing_results = {
                    'type': 'kegg',
                    'data': results,
                    'include_sequences': include_sequences
                }
                st.success("✅ Processing completed!")
                st.rerun()
            else:
                st.error("❌ Processing failed")
                
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")

def process_direct_results(processor, assembly):
    """Process direct search results"""
    with st.spinner("Processing assembly with AI..."):
        try:
            # Download and process
            genbank_data = processor.fetch_genbank(assembly['assembly_acc'])
            if genbank_data:
                # Use AI parser by default
                smart_parser = SmartParserGenerator()
                genes = smart_parser.parse_with_claude(
                    genbank_data, 
                    st.session_state.selected_features
                )
                
                if genes:
                    st.session_state.processing_results = {
                        'type': 'direct',
                        'data': {
                            'genes': genes,
                            'assembly': assembly,
                            'total_genes': len(genes)
                        }
                    }
                    st.success(f"✅ Processed {len(genes)} genes")
                    st.rerun()
                else:
                    st.error("❌ No genes found")
            else:
                st.error("❌ Failed to download GenBank data")
                
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")

def render_processing_results():
    """Display processing results"""
    st.header("📈 Processing Results")
    
    results = st.session_state.processing_results
    
    if results['type'] == 'kegg':
        # KEGG processing results
        data = results['data']
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Processed", data['successful_files'])
        with col2:
            st.metric("Total Genes", data['total_genes'])
        with col3:
            st.metric("Success Rate", f"{data['success_rate']:.1f}%")
        
        # Individual file results
        with st.expander("Individual File Results", expanded=False):
            for file_result in data['individual_results']:
                if file_result['status'] == 'success':
                    st.success(f"✅ {file_result['accession']}: {file_result['gene_count']} genes")
                else:
                    st.error(f"❌ {file_result['accession']}: {file_result.get('error', 'Unknown error')}")
        
        # Prepare data for sequence mapping
        if data['total_genes'] > 0:
            st.session_state.final_results = {
                'type': 'kegg',
                'csv_files': data.get('csv_files', []),
                'gene_data': data.get('combined_data', [])
            }
    
    elif results['type'] == 'direct':
        # Direct processing results
        data = results['data']
        
        # Summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Assembly", data['assembly']['assembly_acc'])
        with col2:
            st.metric("Total Genes", data['total_genes'])
        
        # Gene preview
        if data['genes']:
            st.subheader("Gene Preview")
            df = pd.DataFrame(data['genes'][:10])  # Show first 10 genes
            st.dataframe(df)
            
            # Download button
            if data['genes']:
                csv_data = pd.DataFrame(data['genes']).to_csv(index=False)
                st.download_button(
                    label="📥 Download Gene Data (CSV)",
                    data=csv_data,
                    file_name=f"{data['assembly']['assembly_acc']}_genes.csv",
                    mime="text/csv"
                )
            
            # Prepare for sequence mapping if sequences are included
            if any(gene.get('sequence') for gene in data['genes']):
                st.session_state.final_results = {
                    'type': 'direct',
                    'gene_data': data['genes'],
                    'assembly': data['assembly']
                }

def render_integrated_sequence_mapping():
    """Integrated sequence mapping at the end of the workflow"""
    st.header("🔗 Sequence Analysis & Mapping")
    
    results = st.session_state.final_results
    
    if results['type'] == 'kegg':
        st.info("🧬 Ready for sequence mapping analysis with multiple genomes")
        
        # Check if we have sequence data
        csv_files = results.get('csv_files', [])
        if csv_files:
            st.subheader("Available Files for Mapping")
            
            # File selection
            selected_files = st.multiselect(
                "Select files for sequence mapping:",
                options=csv_files,
                default=csv_files[:2] if len(csv_files) > 1 else csv_files
            )
            
            if len(selected_files) >= 2:
                # Mapping parameters
                col1, col2 = st.columns(2)
                with col1:
                    threshold = st.slider("Similarity Threshold", 0.5, 1.0, 0.8, 0.05)
                with col2:
                    use_clustering = st.checkbox("Use Clustering (Faster)", value=True)
                
                if st.button("🔍 Run Sequence Mapping", type="primary"):
                    run_sequence_mapping(selected_files, threshold, use_clustering)
        else:
            st.warning("⚠️ No sequence data available. Please re-run with 'Include DNA sequences' enabled.")
    
    elif results['type'] == 'direct':
        st.info("🧬 Single genome processed - sequence mapping requires multiple genomes")
        
        # Upload additional files for comparison
        st.subheader("Upload Additional CSV Files for Comparison")
        uploaded_files = st.file_uploader(
            "Upload CSV files with gene sequences:",
            accept_multiple_files=True,
            type=['csv']
        )
        
        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} files")
            # Process uploaded files and run mapping
            if st.button("🔍 Compare with Uploaded Files", type="primary"):
                run_comparison_with_uploads(results['gene_data'], uploaded_files)

def run_sequence_mapping(selected_files, threshold, use_clustering):
    """Run sequence mapping analysis"""
    with st.spinner("Running sequence mapping analysis..."):
        try:
            analyzer = SequenceAnalyzer()
            
            # Load file data
            file_data = {}
            for file_path in selected_files:
                df = pd.read_csv(file_path)
                file_data[os.path.basename(file_path)] = df
            
            # Run analysis
            query_file = list(file_data.keys())[0]
            target_files = list(file_data.keys())[1:]
            
            result = analyzer.run_complete_mapping_analysis(
                query_file, target_files, file_data, threshold, use_clustering
            )
            
            if result['status'] == 'success':
                st.session_state.mapping_results = result
                display_mapping_results(result)
            else:
                st.error(f"❌ Mapping failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Error during mapping: {str(e)}")

def run_comparison_with_uploads(gene_data, uploaded_files):
    """Run comparison with uploaded files"""
    with st.spinner("Processing uploaded files and running comparison..."):
        try:
            # Process uploaded files
            file_data = {}
            
            # Add current gene data
            current_df = pd.DataFrame(gene_data)
            file_data['current_genome'] = current_df
            
            # Process uploaded files
            for uploaded_file in uploaded_files:
                df = pd.read_csv(uploaded_file)
                file_data[uploaded_file.name] = df
            
            # Run mapping
            analyzer = SequenceAnalyzer()
            result = analyzer.run_complete_mapping_analysis(
                'current_genome', 
                list(file_data.keys())[1:], 
                file_data, 
                0.8, 
                True
            )
            
            if result['status'] == 'success':
                st.session_state.mapping_results = result
                display_mapping_results(result)
            else:
                st.error(f"❌ Comparison failed: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"❌ Error during comparison: {str(e)}")

def display_mapping_results(result):
    """Display sequence mapping results"""
    st.subheader("🎯 Mapping Results")
    
    matches = result.get('matches', [])
    if matches:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Matches", len(matches))
        with col2:
            high_confidence = len([m for m in matches if m['Match Score'] > 0.9])
            st.metric("High Confidence", high_confidence)
        with col3:
            avg_score = sum(m['Match Score'] for m in matches) / len(matches)
            st.metric("Average Score", f"{avg_score:.3f}")
        
        # Results table
        df = pd.DataFrame(matches)
        st.dataframe(df, use_container_width=True)
        
        # Download results
        if not df.empty:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Mapping Results",
                data=csv_data,
                file_name="sequence_mapping_results.csv",
                mime="text/csv"
            )
        
        # Visualizations (if available)
        if result.get('show_visualizations'):
            st.subheader("📊 Visualizations")
            
            # Create simple visualization
            if 'Match Score' in df.columns and len(df) > 0:
                try:
                    score_hist = df['Match Score'].hist(bins=10)
                    st.pyplot(score_hist.figure)
                except Exception:
                    st.info("Visualization not available")
    else:
        st.warning("⚠️ No matches found with the current threshold")

if __name__ == "__main__":
    main()