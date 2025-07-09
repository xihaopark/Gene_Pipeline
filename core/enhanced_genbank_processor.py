# core/enhanced_genbank_processor.py
import logging
import time
from typing import List, Dict, Optional
from .genbank_processor import GenBankProcessor
from .smart_parser_generator import SmartParserGenerator
import streamlit as st

class EnhancedGenBankProcessor(GenBankProcessor):
    """Enhanced GenBank processor with smart parsing capabilities"""
    
    def __init__(self):
        super().__init__()
        self.smart_parser = None
        self._init_smart_parser()
        
    def _init_smart_parser(self):
        """Initialize smart parser if API key is available"""
        api_key = st.session_state.get("anthropic_api_key", "") or st.secrets.get("ANTHROPIC_API_KEY", "")
        if api_key:
            self.smart_parser = SmartParserGenerator(api_key)
            logging.info("Smart parser initialized with Claude API")
        else:
            self.smart_parser = SmartParserGenerator()  # Will use fallback parser
            logging.info("Smart parser initialized without Claude API (using fallback)")
    
    def parse_genes_smart(self, genbank_data: str, use_claude: bool = True, 
                         progress_callback: callable = None) -> List[Dict]:
        """Parse genes using smart optimized parser"""
        
        # Update progress
        if progress_callback:
            progress_callback(10, "Analyzing GenBank structure...")
        
        # Generate optimized parser
        if self.smart_parser:
            if progress_callback:
                progress_callback(30, "Generating optimized parser...")
            
            parser_code, metadata = self.smart_parser.generate_optimized_parser(
                genbank_data, use_claude=use_claude
            )
            
            if metadata.get('success'):
                if progress_callback:
                    progress_callback(50, f"Using {metadata['source']} parser...")
                
                # Execute the optimized parser
                namespace = {}
                exec(parser_code, namespace)
                
                if 'parse_genbank_fast' in namespace:
                    if progress_callback:
                        progress_callback(70, "Extracting genes with optimized parser...")
                    
                    start_time = time.time()
                    genes = namespace['parse_genbank_fast'](genbank_data)
                    parse_time = time.time() - start_time
                    
                    logging.info(f"Smart parser extracted {len(genes)} genes in {parse_time:.2f}s")
                    
                    if progress_callback:
                        progress_callback(90, f"Extracted {len(genes)} genes in {parse_time:.2f}s")
                    
                    return genes
        
        # Fallback to BioPython
        if progress_callback:
            progress_callback(50, "Using standard BioPython parser...")
        
        logging.info("Falling back to BioPython parser")
        return self.parse_genes(genbank_data)
    
    def compare_parsing_methods(self, genbank_data: str) -> Dict:
        """Compare performance between BioPython and smart parser"""
        results = {}
        
        # Test BioPython
        start_time = time.time()
        biopython_genes = self.parse_genes(genbank_data)
        biopython_time = time.time() - start_time
        
        results['biopython'] = {
            'time': biopython_time,
            'gene_count': len(biopython_genes),
            'genes_per_second': len(biopython_genes) / biopython_time if biopython_time > 0 else 0
        }
        
        # Test smart parser
        if self.smart_parser:
            parser_code, metadata = self.smart_parser.generate_optimized_parser(genbank_data)
            
            if metadata.get('success'):
                results['smart_parser'] = {
                    'time': metadata.get('parse_time', 0),
                    'gene_count': metadata.get('gene_count', 0),
                    'genes_per_second': metadata.get('genes_per_second', 0),
                    'source': metadata.get('source', 'unknown')
                }
                
                # Calculate speedup
                if results['biopython']['time'] > 0 and results['smart_parser']['time'] > 0:
                    results['speedup'] = results['biopython']['time'] / results['smart_parser']['time']
        
        return results
    
    def parse_genes(self, genbank_data: str, selected_features: List[str] = None) -> List[Dict]:
        """Parse genes from GenBank data with feature selection"""
        if selected_features is None:
            selected_features = ['gene', 'CDS']
        
        # Call parent method but filter by selected features
        all_features = super().parse_genes(genbank_data)
        
        # Filter by selected feature types
        filtered_genes = []
        for gene in all_features:
            if gene.get('feature_type') in selected_features:
                filtered_genes.append(gene)
        
        # Apply additional filters
        min_length = st.session_state.get('min_feature_length', 0)
        if min_length > 0:
            filtered_genes = [g for g in filtered_genes if g.get('length', 0) >= min_length]
        
        return filtered_genes


def render_parser_settings():
    """Render parser settings in the sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Parser Settings")
    
    # API key input
    api_key = st.sidebar.text_input(
        "Claude API Key (optional):",
        value=st.session_state.get("anthropic_api_key", ""),
        type="password",
        help="Enter Claude API key for intelligent parser generation"
    )
    
    if api_key:
        st.session_state.anthropic_api_key = api_key
    
    # Parser mode selection
    parser_mode = st.sidebar.radio(
        "Parser Mode:",
        ["Auto (Recommended)", "Smart Parser Only", "BioPython Only"],
        help="Auto mode tries smart parser first, then falls back to BioPython"
    )
    
    st.session_state.parser_mode = parser_mode
    
    # Performance comparison toggle
    show_performance = st.sidebar.checkbox(
        "Show Performance Metrics",
        value=st.session_state.get("show_performance", False),
        help="Display parsing performance comparison"
    )
    
    st.session_state.show_performance = show_performance
    
    return {
        'api_key': api_key,
        'parser_mode': parser_mode,
        'show_performance': show_performance
    }


def display_performance_metrics(metrics: Dict):
    """Display parser performance metrics"""
    if not metrics:
        return
    
    st.subheader("⚡ Parser Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    if 'biopython' in metrics:
        with col1:
            st.metric(
                "BioPython Parser",
                f"{metrics['biopython']['time']:.2f}s",
                f"{metrics['biopython']['genes_per_second']:.0f} genes/s"
            )
    
    if 'smart_parser' in metrics:
        with col2:
            st.metric(
                f"Smart Parser ({metrics['smart_parser']['source']})",
                f"{metrics['smart_parser']['time']:.2f}s",
                f"{metrics['smart_parser']['genes_per_second']:.0f} genes/s"
            )
    
    if 'speedup' in metrics:
        with col3:
            speedup = metrics['speedup']
            color = "green" if speedup > 1 else "red"
            st.metric(
                "Speedup",
                f"{speedup:.1f}x",
                delta=f"{(speedup - 1) * 100:.0f}% faster" if speedup > 1 else f"{(1 - speedup) * 100:.0f}% slower"
            )