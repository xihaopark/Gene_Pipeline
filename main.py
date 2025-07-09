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
    
    # Initialize default settings
    if 'parser_method' not in st.session_state:
        st.session_state.parser_method = "BioPython (Default)"
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = ['gene', 'CDS']
    
    st.title("GenBank Gene Extractor")
    st.write("Hello! Please select your desired function from the tabs below.")
    
    # Initialize processors
    processor = GenBankProcessor()
    kegg_integration = KEGGIntegration()
    staged_processor = StagedGenomeProcessor()
    
    # Main content area - Three primary tabs
    tab1, tab2, tab3 = st.tabs([
        "KEGG Search", 
        "Direct Search",
        "Sequence Mapping"
    ])
    
    with tab1:
        render_kegg_search_tab(staged_processor)
    
    with tab2:
        render_direct_search_tab(processor)
    
    with tab3:
        render_sequence_mapping_tab()

def render_kegg_search_tab(staged_processor):
    """KEGG batch search and processing"""
    st.header("KEGG Organism Search")
    
    # Parser options in expandable section
    with st.expander("Parser Options", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            parser_method = st.radio(
                "Select parser:",
                ["BioPython (Default)", "Claude AI (Fast)"]
            )
            st.session_state.parser_method = parser_method
        
        with col2:
            selected_features = st.multiselect(
                "Features to extract:",
                options=['gene', 'CDS', 'mRNA', 'tRNA', 'rRNA', 'misc_RNA'],
                default=st.session_state.get('selected_features', ['gene', 'CDS'])
            )
            st.session_state.selected_features = selected_features
    
    # Input area
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        organism_name = st.text_input(
            "Organism Name:",
            value=st.session_state.get("kegg_organism", ""),
            placeholder="e.g., Clostridioides difficile"
        )
    
    with col2:
        max_genomes = st.number_input(
            "Max Genomes:",
            min_value=1,
            max_value=20,
            value=5
        )
    
    with col3:
        include_sequences = st.checkbox(
            "Include Sequences",
            value=True
        )
    
    if organism_name:
        st.session_state.kegg_organism = organism_name
        
        # Stage 1: Download
        if st.button("Download GenBank Files", type="primary"):
            run_stage1_download(staged_processor, organism_name, max_genomes)
        
        # Stage 2: Preview (if download results exist)
        if st.session_state.get("download_results"):
            st.markdown("---")
            render_stage2_preview()
            
            # Stage 3: Parse
            if st.button("Parse Files", type="primary"):
                run_stage3_parsing(staged_processor, include_sequences)
        
        # Results display
        if st.session_state.get("parsing_results"):
            render_batch_results()

def render_direct_search_tab(processor):
    """Direct genome accession search"""
    st.header("Direct Accession Search")
    
    accession = st.text_input(
        "Genome Accession Number:",
        value=st.session_state.get("accession", ""),
        placeholder="e.g., GCA_000005825.2"
    )
    
    if accession:
        st.session_state.accession = accession
        
        if st.button("Search Genome", type="primary"):
            with st.spinner("Searching..."):
                try:
                    assemblies = processor.search_assembly(accession)
                    if assemblies:
                        st.session_state.assemblies = assemblies
                        ImprovedStateManager.update_processing_stage("ready", {"assemblies": assemblies})
                    else:
                        st.error("No assemblies found")
                except Exception:
                    st.error("Search error")
        
        # Display search results
        if st.session_state.get("assemblies"):
            st.subheader("Search Results")
            for i, asm in enumerate(st.session_state.assemblies):
                with st.expander(f"{asm['assembly_acc']} - {asm['organism']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Assembly Name:** {asm.get('assembly_name', 'N/A')}")
                        st.write(f"**Level:** {asm.get('assembly_level', 'N/A')}")
                    with col2:
                        st.write(f"**Submitter:** {asm.get('submitter', 'N/A')}")
                        st.write(f"**Version:** {asm.get('version', 'N/A')}")
                    
                    if st.button(f"Select Assembly {i+1}", key=f"select_{i}"):
                        st.session_state.selected_assembly = asm
                        st.rerun()
        
        # Process selected assembly
        if st.session_state.get("selected_assembly"):
            selected = st.session_state.selected_assembly
            st.success(f"Selected: {selected.get('assembly_acc', 'Unknown')}")
            
            if st.button("Download and Extract Genes", type="primary"):
                process_single_assembly(processor, selected)

def run_stage1_download(staged_processor, organism_name, max_genomes):
    """Execute stage 1 download with progress tracking"""
    ImprovedStateManager.update_processing_stage("downloading")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress: int, status: str):
        progress_bar.progress(progress / 100)
        status_text.text(status)
    
    try:
        download_results = staged_processor.stage1_download_all_genbank_files(
            organism_name, max_genomes, progress_callback=update_progress
        )
        
        st.session_state.download_results = download_results
        
        if download_results['status'] == 'completed':
            progress_bar.empty()
            status_text.empty()
            st.success(f"Downloaded {download_results['successful_downloads']} files")
            ImprovedStateManager.update_processing_stage("ready")
        else:
            st.error("Download failed")
            ImprovedStateManager.update_processing_stage("error")
            
    except Exception:
        st.error("Error during download")
        ImprovedStateManager.update_processing_stage("error")

def render_stage2_preview():
    """Render stage 2 preview"""
    download_results = st.session_state.download_results
    
    st.subheader("Download Results")
    
    # Generate summary report
    summary = StagedGenomeProcessor().stage2_generate_summary_report(download_results)
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Organism", summary['organism'])
    with col2:
        st.metric("Downloaded", f"{summary['successful_files']}/{summary['total_files']}")
    with col3:
        st.metric("Total Genes", summary['total_genes'])
    with col4:
        st.metric("File Size", summary['total_size_mb'])

def run_stage3_parsing(staged_processor, include_sequences):
    """Execute stage 3 parsing"""
    download_results = st.session_state.download_results
    
    ImprovedStateManager.update_processing_stage("parsing")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress: int, status: str):
        progress_bar.progress(progress / 100)
        status_text.text("Parsing files...")
    
    try:
        parsing_results = staged_processor.stage3_batch_parse_files(
            download_results, include_sequences, progress_callback=update_progress
        )
        
        st.session_state.parsing_results = parsing_results
        
        if parsing_results['status'] == 'completed':
            progress_bar.empty()
            status_text.empty()
            st.success(f"Parsed {parsing_results['total_genes']} genes")
            ImprovedStateManager.update_processing_stage("completed")
        else:
            st.error("Parsing failed")
            ImprovedStateManager.update_processing_stage("error")
            
    except Exception:
        st.error("Error during parsing")
        ImprovedStateManager.update_processing_stage("error")

def render_batch_results():
    """Render batch processing results"""
    parsing_results = st.session_state.parsing_results
    
    st.subheader("Results")
    
    # Overall statistics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Genes", parsing_results['total_genes'])
    with col2:
        st.metric("Files Processed", len(parsing_results['individual_results']))
    
    # Individual file results
    all_csv_files = []
    for accession, result in parsing_results['individual_results'].items():
        with st.expander(f"{accession} - {result['gene_count']} genes"):
            # Preview first 10 genes
            if result['genes']:
                preview_df = pd.DataFrame(result['genes'][:10])
                st.dataframe(preview_df, use_container_width=True)
            
            # Download button
            try:
                with open(result['csv_file'], 'r', encoding='utf-8') as f:
                    csv_content = f.read()
                
                st.download_button(
                    label="Download CSV",
                    data=csv_content,
                    file_name=f"{accession}_genes.csv",
                    mime="text/csv",
                    key=f"download_{accession}"
                )
                
                # Store for batch download
                all_csv_files.append({
                    'accession': accession,
                    'filename': f"{accession}_genes.csv",
                    'content': csv_content
                })
            except Exception:
                pass
    
    # Batch download all CSV files
    st.markdown("---")
    if all_csv_files:
        st.subheader("Batch Download")
        
        # Create a ZIP file with all CSVs
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_info in all_csv_files:
                zip_file.writestr(file_info['filename'], file_info['content'])
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="Download All CSV Files (ZIP)",
            data=zip_buffer,
            file_name=f"{st.session_state.kegg_organism.replace(' ', '_')}_all_genes.zip",
            mime="application/zip"
        )
        
        # Store files in session state for mapping
        st.session_state.batch_csv_files = all_csv_files
    
    # Inline sequence mapping section
    st.markdown("---")
    st.subheader("Sequence Mapping Analysis")
    
    if st.checkbox("Show Sequence Mapping", value=False):
        render_inline_sequence_mapping(all_csv_files)

def process_single_assembly(processor, selected):
    """Process single assembly with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Processing...")
        progress_bar.progress(10)
        
        nuccore_records = processor.link_to_nuccore(selected['assembly_acc'])
        if not nuccore_records:
            progress_bar.empty()
            status_text.empty()
            st.error("No nucleotide sequences found")
            return
        
        best_record = max(nuccore_records, key=lambda x: x.get('length', 0))
        
        progress_bar.progress(30)
        
        genbank_data = processor.fetch_genbank(best_record['accver'], save_to_cache=True)
        if not genbank_data:
            progress_bar.empty()
            status_text.empty()
            st.error("Failed to download GenBank data")
            return
        
        progress_bar.progress(60)
        
        genes = processor.parse_genes(genbank_data)
        
        if genes:
            progress_bar.progress(90)
            
            st.session_state.genes = genes
            st.session_state.genbank_accession = best_record['accver']
            
            progress_bar.progress(100)
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"Extracted {len(genes)} genes")
            ImprovedStateManager.update_processing_stage("completed")
            
            # Display results
            df = pd.DataFrame(genes[:10])
            st.dataframe(df, use_container_width=True)
            
            # Download CSV
            csv_content = processor.genes_to_csv(genes, best_record['accver'])
            st.download_button(
                label="Download CSV",
                data=csv_content,
                file_name=f"{best_record['accver']}_genes.csv",
                mime="text/csv"
            )
            
            # Store for mapping
            st.session_state.single_csv_file = {
                'accession': best_record['accver'],
                'filename': f"{best_record['accver']}_genes.csv",
                'content': csv_content
            }
            
            # Note about sequence mapping
            st.markdown("---")
            st.info("For sequence mapping with other genomes, use the Sequence Mapping tab")
        else:
            progress_bar.empty()
            status_text.empty()
            st.error("No genes found")
            
    except Exception:
        progress_bar.empty()
        status_text.empty()
        st.error("Processing error")

def render_sequence_mapping_tab():
    """Render sequence mapping analysis tab"""
    st.header("Sequence Mapping Analysis")
    
    # Prepare file data
    file_data = {}
    valid_files = []
    
    # Check if we have files from previous tabs
    has_batch_files = 'batch_csv_files' in st.session_state and st.session_state.batch_csv_files
    has_single_file = 'single_csv_file' in st.session_state and st.session_state.single_csv_file
    
    if has_batch_files or has_single_file:
        st.info("Files from previous analysis detected. You can use them directly or upload new files.")
        
        # Load files from previous analysis
        if has_batch_files:
            for file_info in st.session_state.batch_csv_files:
                df = pd.read_csv(StringIO(file_info['content']))
                file_data[file_info['filename']] = df
                valid_files.append({
                    'name': file_info['filename'],
                    'genes': len(df)
                })
        
        elif has_single_file:
            file_info = st.session_state.single_csv_file
            df = pd.read_csv(StringIO(file_info['content']))
            file_data[file_info['filename']] = df
            valid_files.append({
                'name': file_info['filename'],
                'genes': len(df)
            })
    
    # File upload section
    st.subheader("Upload Additional Files")
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        accept_multiple_files=True,
        type=['csv'],
        key="local_mapping_files"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Locus Tag' in df.columns and 'NT Seq' in df.columns:
                    has_sequences = df['NT Seq'].notna().any() and (df['NT Seq'] != '').any()
                    if has_sequences:
                        file_data[uploaded_file.name] = df
                        valid_files.append({
                            'name': uploaded_file.name,
                            'genes': len(df)
                        })
                        st.success(f"{uploaded_file.name} - {len(df)} genes")
            except Exception:
                st.error(f"{uploaded_file.name} - Error reading file")
    
    # Display loaded files
    if valid_files:
        st.subheader("Loaded Files")
        for file_info in valid_files:
            st.write(f"- {file_info['name']} ({file_info['genes']} genes)")
    
    # Mapping configuration
    if len(valid_files) >= 2:
        st.markdown("---")
        st.subheader("Configure Mapping")
        
        # Advanced options
        with st.expander("Advanced Options", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                use_clustering = st.checkbox("Use Clustering", value=True, 
                                           help="Use clustering to speed up matching")
            
            with col2:
                if use_clustering:
                    n_clusters = st.number_input("Number of Clusters", 
                                               min_value=5, max_value=50, value=20,
                                               help="Number of clusters for grouping sequences")
                else:
                    n_clusters = None
            
            with col3:
                show_visualizations = st.checkbox("Show Visualizations", value=True,
                                                help="Generate clustering and match visualizations")
        
        # File selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            query_file = st.selectbox(
                "Query file:",
                options=[f['name'] for f in valid_files]
            )
        
        with col2:
            available_targets = [f['name'] for f in valid_files if f['name'] != query_file]
            target_files = st.multiselect(
                "Target files:",
                options=available_targets,
                default=available_targets[:1] if available_targets else []
            )
        
        with col3:
            match_threshold = st.slider(
                "Match threshold:",
                min_value=0.7,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Minimum similarity for a match (0.7 = 70%)"
            )
        
        # Run analysis
        if query_file and target_files:
            if st.button("Run Mapping Analysis", type="primary"):
                analyzer = SequenceAnalyzer()
                
                with st.spinner("Running mapping analysis..."):
                    result = analyzer.run_complete_mapping_analysis(
                        query_file, target_files, file_data, match_threshold,
                        use_clustering, n_clusters if use_clustering else 20,
                        show_visualizations
                    )
                
                # Display results
                if result['status'] == 'success':
                    display_mapping_results(result, analyzer)
                else:
                    st.error(result.get('message', 'Mapping failed'))
    
    elif len(valid_files) == 1:
        st.warning("Please provide at least 2 files for comparison")
    else:
        st.info("Please upload or load gene table files to begin mapping analysis")

def render_inline_sequence_mapping(csv_files):
    """Render inline sequence mapping for batch results"""
    
    if not csv_files or len(csv_files) < 2:
        st.warning("Need at least 2 files for sequence mapping")
        return
    
    # Convert to dataframes
    file_data = {}
    valid_files = []
    
    for file_info in csv_files:
        df = pd.read_csv(StringIO(file_info['content']))
        
        # Check if has sequences
        has_sequences = 'NT Seq' in df.columns and df['NT Seq'].notna().any()
        
        if has_sequences:
            file_data[file_info['filename']] = df
            valid_files.append({
                'name': file_info['filename'],
                'genes': len(df)
            })
    
    if len(valid_files) < 2:
        st.warning("Need at least 2 files with sequences for mapping")
        return
    
    # Use the same mapping interface
    analyzer = SequenceAnalyzer()
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        query_file = st.selectbox(
            "Query file:",
            options=[f['name'] for f in valid_files],
            key="inline_query"
        )
    
    with col2:
        available_targets = [f['name'] for f in valid_files if f['name'] != query_file]
        target_files = st.multiselect(
            "Target files:",
            options=available_targets,
            default=available_targets[:1],
            key="inline_targets"
        )
    
    with col3:
        match_threshold = st.slider(
            "Match threshold:",
            min_value=0.7,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="inline_threshold"
        )
    
    if query_file and target_files:
        if st.button("Run Quick Mapping", type="primary", key="inline_run"):
            with st.spinner("Running mapping analysis..."):
                result = analyzer.run_complete_mapping_analysis(
                    query_file, target_files, file_data, match_threshold,
                    use_clustering=True, n_clusters=20, show_visualizations=False
                )
            
            if result['status'] == 'success':
                display_mapping_results(result, analyzer)
            else:
                st.error(result.get('message', 'Mapping failed'))

def display_mapping_results(result: Dict, analyzer: SequenceAnalyzer):
    """Display mapping analysis results"""
    
    if not result.get('results_df').empty:
        results_df = result['results_df']
        
        # Summary statistics
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Matches", len(results_df))
        
        with col2:
            avg_score = results_df['Match Score'].mean()
            st.metric("Average Score", f"{avg_score:.3f}")
        
        with col3:
            if 'saved_comparisons' in result:
                st.metric("Comparisons Saved", f"{result['saved_comparisons']:,}")
            else:
                st.metric("Total Comparisons", f"{result['total_comparisons']:,}")
        
        # Visualizations
        if result.get('show_visualizations') and 'cluster_result' in result:
            st.markdown("---")
            st.subheader("Visualizations")
            
            tab1, tab2, tab3, tab4 = st.tabs(["Cluster Map", "Match Heatmap", "Sankey Diagram", "Cluster Distribution"])
            
            with tab1:
                fig_cluster = analyzer.create_cluster_visualization(result['cluster_result'])
                st.plotly_chart(fig_cluster, use_container_width=True)
            
            with tab2:
                query_files = list(set(results_df['Query File'].unique()))
                target_files = list(set(results_df['Target File'].unique()))
                fig_heatmap = analyzer.create_heatmap(result['matches'], query_files, target_files)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with tab3:
                fig_sankey = analyzer.create_sankey_diagram(result['matches'], result['all_files'])
                st.plotly_chart(fig_sankey, use_container_width=True)
            
            with tab4:
                fig_enrichment = analyzer.create_enrichment_plot(result['cluster_result'])
                st.plotly_chart(fig_enrichment, use_container_width=True)
        
        # Results table
        st.markdown("---")
        st.subheader("Match Results")
        
        # Group by target file
        for target_file in results_df['Target File'].unique():
            target_matches = results_df[results_df['Target File'] == target_file]
            with st.expander(f"{target_file} ({len(target_matches)} matches)"):
                st.dataframe(target_matches, use_container_width=True)
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="mapping_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("No matches found above the threshold")

if __name__ == "__main__":
    main()