# core/feature_selector.py
import streamlit as st
from typing import List, Dict, Set

class FeatureSelector:
    """Handle feature selection for GenBank parsing"""
    
    # Common GenBank feature types
    AVAILABLE_FEATURES = {
        'gene': 'Gene features',
        'CDS': 'Coding sequences', 
        'mRNA': 'mRNA features',
        'tRNA': 'Transfer RNA',
        'rRNA': 'Ribosomal RNA',
        'misc_RNA': 'Miscellaneous RNA',
        'regulatory': 'Regulatory elements',
        'repeat_region': 'Repeat regions',
        'mobile_element': 'Mobile genetic elements',
        'source': 'Source information',
        'misc_feature': 'Miscellaneous features'
    }
    
    # Default features for gene analysis
    DEFAULT_FEATURES = ['gene', 'CDS']
    
    @staticmethod
    def render_feature_selector() -> List[str]:
        """Render feature selection UI in sidebar"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Feature Selection")
        
        # Quick selection buttons
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Genes Only"):
                st.session_state.selected_features = ['gene', 'CDS']
                st.rerun()
        with col2:
            if st.button("All RNA"):
                st.session_state.selected_features = ['mRNA', 'tRNA', 'rRNA', 'misc_RNA']
                st.rerun()
        
        # Initialize selected features
        if 'selected_features' not in st.session_state:
            st.session_state.selected_features = FeatureSelector.DEFAULT_FEATURES
        
        # Multi-select for features
        selected = st.sidebar.multiselect(
            "Select features to extract:",
            options=list(FeatureSelector.AVAILABLE_FEATURES.keys()),
            default=st.session_state.selected_features,
            format_func=lambda x: f"{x} - {FeatureSelector.AVAILABLE_FEATURES[x]}",
            help="Choose which GenBank features to extract"
        )
        
        st.session_state.selected_features = selected
        
        # Additional options
        st.sidebar.markdown("**Additional Options:**")
        
        extract_intergenic = st.sidebar.checkbox(
            "Extract intergenic regions",
            value=st.session_state.get('extract_intergenic', False),
            help="Extract regions between genes"
        )
        st.session_state.extract_intergenic = extract_intergenic
        
        min_feature_length = st.sidebar.number_input(
            "Minimum feature length (bp):",
            min_value=0,
            max_value=10000,
            value=st.session_state.get('min_feature_length', 0),
            step=100,
            help="Filter out features shorter than this length"
        )
        st.session_state.min_feature_length = min_feature_length
        
        return selected
    
    @staticmethod
    def get_selected_features() -> List[str]:
        """Get currently selected features"""
        return st.session_state.get('selected_features', FeatureSelector.DEFAULT_FEATURES)
    
    @staticmethod
    def filter_features_for_parsing(features: List[str]) -> str:
        """Generate filter code for parser"""
        feature_str = '", "'.join(features)
        return f'["{feature_str}"]'


class SequenceMappingFormatter:
    """Format parsed data for sequence mapping compatibility"""
    
    @staticmethod
    def format_for_mapping(genes: List[Dict], accession: str) -> List[Dict]:
        """Format gene data to match the expected input format for Info Mapping.ipynb"""
        formatted_data = []
        
        for gene in genes:
            # Ensure we have the required fields for mapping
            formatted_gene = {
                'Locus Tag': gene.get('locus_tag', ''),
                'Gene': gene.get('gene', 'N/A'),
                'EC Number': gene.get('ec_number', 'N/A'),
                'NT Seq': gene.get('sequence', ''),  # Nucleotide sequence
                'Product': gene.get('product', ''),
                'Start': gene.get('start', 0),
                'End': gene.get('end', 0),
                'Strand': gene.get('strand', '+'),
                'Feature Type': gene.get('feature_type', 'gene')
            }
            
            # Only include if we have both locus tag and sequence
            if formatted_gene['Locus Tag'] and formatted_gene['NT Seq']:
                formatted_data.append(formatted_gene)
        
        return formatted_data
    
    @staticmethod
    def validate_mapping_format(df) -> Dict[str, bool]:
        """Validate if dataframe has required columns for mapping"""
        required_columns = ['Locus Tag', 'Gene', 'EC Number', 'NT Seq']
        
        validation = {
            'has_all_required': all(col in df.columns for col in required_columns),
            'has_locus_tag': 'Locus Tag' in df.columns,
            'has_gene': 'Gene' in df.columns,
            'has_ec_number': 'EC Number' in df.columns,
            'has_nt_seq': 'NT Seq' in df.columns,
            'non_empty_sequences': False
        }
        
        if 'NT Seq' in df.columns:
            validation['non_empty_sequences'] = df['NT Seq'].notna().any() and (df['NT Seq'] != '').any()
        
        return validation
    
    @staticmethod
    def generate_mapping_ready_csv(genes: List[Dict], output_path: str, accession: str) -> bool:
        """Generate CSV file ready for sequence mapping analysis"""
        import pandas as pd
        
        try:
            # Format data for mapping
            formatted_data = SequenceMappingFormatter.format_for_mapping(genes, accession)
            
            if not formatted_data:
                return False
            
            # Create DataFrame with specific column order
            df = pd.DataFrame(formatted_data)
            
            # Ensure column order matches expected format
            column_order = ['Locus Tag', 'Gene', 'EC Number', 'NT Seq', 'Product', 
                          'Start', 'End', 'Strand', 'Feature Type']
            
            # Only include columns that exist
            columns_to_use = [col for col in column_order if col in df.columns]
            df = df[columns_to_use]
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            
            return True
            
        except Exception as e:
            logging.error(f"Error generating mapping-ready CSV: {e}")
            return False
    
    @staticmethod
    def render_mapping_export_ui(genes: List[Dict], accession: str):
        """Render UI for exporting mapping-compatible files"""
        st.subheader("🔄 Export for Sequence Mapping")
        
        # Validate current data
        if not genes:
            st.warning("No gene data available for export")
            return
        
        # Check if sequences are included
        has_sequences = any(gene.get('sequence') for gene in genes)
        
        if not has_sequences:
            st.error("⚠️ No sequences found! Please re-parse with 'Include Sequences' option enabled.")
            st.info("The sequence mapping tool requires nucleotide sequences (NT Seq) for comparison.")
            return
        
        # Format preview
        formatted_data = SequenceMappingFormatter.format_for_mapping(genes, accession)
        
        if formatted_data:
            st.success(f"✅ {len(formatted_data)} genes ready for sequence mapping")
            
            # Preview
            preview_df = pd.DataFrame(formatted_data[:5])
            st.write("**Preview (first 5 genes):**")
            st.dataframe(preview_df[['Locus Tag', 'Gene', 'EC Number', 'NT Seq']].assign(
                NT_Seq=preview_df['NT Seq'].str[:50] + '...'
            ))
            
            # Download button
            csv_content = pd.DataFrame(formatted_data).to_csv(index=False)
            
            st.download_button(
                label="📥 Download Mapping-Ready CSV",
                data=csv_content,
                file_name=f"{accession}_mapping_ready.csv",
                mime="text/csv",
                help="Download CSV formatted for sequence mapping analysis"
            )
            
            # Instructions
            with st.expander("📖 How to use with Info Mapping tool"):
                st.markdown("""
                1. Download this CSV file
                2. Place it in the 'Input' folder of your mapping tool
                3. Run the mapping analysis with other genome files
                4. The tool will compare sequences and generate similarity scores
                
                **Required columns:**
                - `Locus Tag`: Unique identifier for each gene
                - `Gene`: Gene name
                - `EC Number`: Enzyme commission number
                - `NT Seq`: Nucleotide sequence (required for comparison)
                """)
        else:
            st.error("Failed to format data for mapping")