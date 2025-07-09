# core/staged_processor.py
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Callable
from .genbank_processor import GenBankProcessor
from .kegg_integration import KEGGIntegration
from .folder_manager import FolderManager
from .smart_parser_generator import SmartParserGenerator
import streamlit as st
import pandas as pd
from io import StringIO

class StagedGenomeProcessor:
    """Three-stage genome processing: Download -> Preview -> Parse"""
    
    def __init__(self):
        self.genbank_processor = GenBankProcessor()
        self.kegg_integration = KEGGIntegration()
        self.folder_manager = FolderManager()
        self.smart_parser = SmartParserGenerator()
        
    def stage1_download_all_genbank_files(self, organism_name: str, max_genomes: int = 10, 
                                         progress_callback: Callable = None) -> Dict:
        """Stage 1: Download all GenBank files to local storage with progress updates"""
        logging.info(f"Stage 1: Starting download for {organism_name}")
        
        # Get KEGG genome information
        genomes, genbank_mapping = self.kegg_integration.search_and_extract_all(organism_name)
        
        if not genomes:
            return {
                'status': 'failed',
                'message': f'No genomes found for {organism_name}',
                'files': []
            }
        
        # Limit processing count
        if len(genomes) > max_genomes:
            genomes = genomes[:max_genomes]
            logging.info(f"Limited to {max_genomes} genomes")
        
        # Collect all GenBank accessions, filter out Assembly accessions
        all_accessions = []
        for genome in genomes:
            t_number = genome['t_number']
            if t_number in genbank_mapping:
                accessions = genbank_mapping[t_number]['genbank_accessions']
                # Filter out Assembly accessions, keep only sequence accessions
                sequence_accessions = [acc for acc in accessions 
                                     if not acc.startswith(('GCA_', 'GCF_'))]
                all_accessions.extend(sequence_accessions)
        
        # Remove duplicates
        unique_accessions = list(set(all_accessions))
        logging.info(f"Found {len(unique_accessions)} unique sequence accessions to download")
        
        # Create project folder
        project_name = st.session_state.get("project_name", organism_name.replace(" ", "_"))
        project_folder = self.folder_manager.create_project_folder(project_name)
        genbank_folder = Path(project_folder) / "genbank_files"
        
        # Download all files
        download_results = {}
        successful_downloads = []
        
        for i, accession in enumerate(unique_accessions):
            # Update progress
            progress = int(((i + 1) / len(unique_accessions)) * 100)
            if progress_callback:
                progress_callback(progress, f"Downloading {accession}... ({i+1}/{len(unique_accessions)})")
            
            logging.info(f"Downloading {i+1}/{len(unique_accessions)}: {accession}")
            
            try:
                # Download GenBank data
                genbank_data = self.genbank_processor.fetch_genbank(accession, save_to_cache=False)
                
                if genbank_data:
                    # Save to local file
                    file_path = genbank_folder / f"{accession}.gb"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(genbank_data)
                    
                    # Analyze basic file information
                    file_info = self._analyze_genbank_file(genbank_data, accession)
                    file_info['file_path'] = str(file_path)
                    file_info['file_size'] = len(genbank_data)
                    
                    download_results[accession] = {
                        'status': 'success',
                        'info': file_info,
                        'file_path': str(file_path)
                    }
                    successful_downloads.append(accession)
                    
                    logging.info(f"✅ {accession}: Downloaded and saved ({file_info['gene_count']} genes)")
                else:
                    download_results[accession] = {
                        'status': 'download_failed',
                        'message': 'Failed to download GenBank data'
                    }
                    logging.error(f"❌ {accession}: Download failed")
            
            except Exception as e:
                download_results[accession] = {
                    'status': 'error',
                    'message': str(e)
                }
                logging.error(f"❌ {accession}: Error - {e}")
            
            time.sleep(0.3)  # Avoid API rate limits
        
        return {
            'status': 'completed' if successful_downloads else 'failed',
            'total_requested': len(unique_accessions),
            'successful_downloads': len(successful_downloads),
            'results': download_results,
            'project_folder': project_folder,
            'organism': organism_name,
            'genomes_info': genomes
        }
    
    def _analyze_genbank_file(self, genbank_data: str, accession: str) -> Dict:
        """Quick analysis of GenBank file basic information"""
        lines = genbank_data.split('\n')
        
        info = {
            'accession': accession,
            'organism': '',
            'length': 0,
            'gene_count': 0,
            'cds_count': 0,
            'has_features': False,
            'definition': ''
        }
        
        # Check first 50 lines for basic info
        for line in lines[:50]:
            line = line.strip()
            
            if line.startswith('DEFINITION'):
                info['definition'] = line.replace('DEFINITION', '').strip()
            elif line.startswith('ORGANISM'):
                info['organism'] = line.replace('ORGANISM', '').strip()
            elif line.startswith('LOCUS'):
                # Extract sequence length
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        info['length'] = int(part)
                        break
            elif line.startswith('FEATURES'):
                info['has_features'] = True
        
        # Quick gene count calculation
        if info['has_features']:
            info['gene_count'] = genbank_data.count('     gene            ')
            info['cds_count'] = genbank_data.count('     CDS             ')
        
        return info
    
    def stage2_generate_summary_report(self, download_results: Dict) -> Dict:
        """Stage 2: Generate download summary report"""
        successful_files = [acc for acc, result in download_results['results'].items() 
                           if result['status'] == 'success']
        
        summary = {
            'organism': download_results['organism'],
            'total_files': download_results['total_requested'],
            'successful_files': len(successful_files),
            'success_rate': f"{(len(successful_files) / download_results['total_requested'] * 100):.1f}%",
            'files_detail': [],
            'total_genes': 0,
            'total_size': 0
        }
        
        for accession in successful_files:
            file_info = download_results['results'][accession]['info']
            summary['files_detail'].append(file_info)
            summary['total_genes'] += file_info['gene_count']
            summary['total_size'] += file_info['file_size']
        
        # Format file size
        summary['total_size_mb'] = f"{summary['total_size'] / (1024*1024):.1f} MB"
        
        return summary
    
    def stage3_batch_parse_files(self, download_results: Dict, include_sequences: bool = False,
                                progress_callback: Callable = None) -> Dict:
        """Stage 3: Batch parse files independently with detailed progress updates"""
        successful_files = [acc for acc, result in download_results['results'].items() 
                           if result['status'] == 'success']
        
        # Store results for each file separately
        individual_results = {}
        parsing_results = {}
        
        total_files = len(successful_files)
        
        # Create a container for detailed progress display
        if 'progress_container' in st.session_state:
            progress_container = st.session_state.progress_container
        else:
            progress_container = None
        
        # Initialize statistics
        total_genes_found = 0
        total_parse_time = 0
        
        # Create output folder
        output_folder = Path(download_results['project_folder']) / "output"
        output_folder.mkdir(exist_ok=True)
        
        for i, accession in enumerate(successful_files):
            file_start_time = time.time()
            
            # Calculate overall progress
            overall_progress = int((i / total_files) * 100)
            
            # Update main status
            if progress_callback:
                progress_callback(overall_progress, f"📂 Processing file {i+1}/{total_files}: {accession}")
            
            logging.info(f"Parsing {i+1}/{total_files}: {accession}")
            
            try:
                file_path = download_results['results'][accession]['file_path']
                
                # Read file
                with open(file_path, 'r', encoding='utf-8') as f:
                    genbank_data = f.read()
                
                # Get parser method selection
                parser_method = st.session_state.get("parser_method", "BioPython (Default)")
                selected_features = st.session_state.get("selected_features", ['gene', 'CDS'])
                
                # Create a sub-progress callback for detailed parsing progress
                def parsing_progress_callback(sub_progress: int, sub_status: str):
                    # Calculate combined progress
                    file_progress = i / total_files
                    current_file_progress = sub_progress / 100.0 / total_files
                    combined_progress = int((file_progress + current_file_progress) * 100)
                    
                    # Update with detailed status
                    detailed_status = f"""
📂 File {i+1}/{total_files}: {accession}
🔄 {sub_status}
📊 Total genes found so far: {total_genes_found}
⏱️ Total parse time: {total_parse_time:.1f}s
                    """.strip()
                    
                    if progress_callback:
                        progress_callback(combined_progress, detailed_status)
                
                # Parse genes based on selected method
                if parser_method == "Claude AI (Fast)":
                    # Show Claude parser generation status
                    if progress_callback:
                        progress_callback(overall_progress, f"🤖 Generating Claude parser for {accession}...")
                    
                    genes = self.smart_parser.parse_with_claude(
                        genbank_data,
                        selected_features,
                        progress_callback=parsing_progress_callback
                    )
                else:
                    # Use BioPython
                    genes = self.smart_parser.parse_with_biopython(
                        genbank_data,
                        selected_features,
                        progress_callback=parsing_progress_callback
                    )
                
                # Calculate parsing time
                file_parse_time = time.time() - file_start_time
                total_parse_time += file_parse_time
                
                if genes:
                    # Format genes for mapping compatibility
                    formatted_genes = []
                    for gene in genes:
                        formatted_gene = {
                            'Locus Tag': gene.get('Locus Tag', gene.get('locus_tag', '')),
                            'Gene': gene.get('Gene', gene.get('gene', 'N/A')),
                            'EC Number': gene.get('EC Number', gene.get('ec_number', 'N/A')),
                            'NT Seq': gene.get('NT Seq', gene.get('sequence', '')) if include_sequences else '',
                            'Product': gene.get('Product', gene.get('product', '')),
                            'Start': gene.get('Start', gene.get('start', 0)),
                            'End': gene.get('End', gene.get('end', 0)),
                            'Strand': gene.get('Strand', gene.get('strand', '+')),
                            'Feature Type': gene.get('Feature Type', gene.get('feature_type', 'gene'))
                        }
                        formatted_genes.append(formatted_gene)
                    
                    # Save individual file results
                    individual_csv_file = output_folder / f"{accession}_genes.csv"
                    df = pd.DataFrame(formatted_genes)
                    df.to_csv(individual_csv_file, index=False)
                    
                    # Store results
                    individual_results[accession] = {
                        'genes': formatted_genes,
                        'gene_count': len(genes),
                        'csv_file': str(individual_csv_file),
                        'has_sequences': any(g.get('NT Seq', '') for g in formatted_genes)
                    }
                    
                    total_genes_found += len(genes)
                    
                    parsing_results[accession] = {
                        'status': 'success',
                        'gene_count': len(genes),
                        'parse_time': file_parse_time,
                        'has_sequences': individual_results[accession]['has_sequences']
                    }
                    
                    # Update status with detailed results
                    if progress_callback:
                        status_msg = f"""
✅ Completed {accession}
📊 Found {len(genes)} genes in {file_parse_time:.2f}s
⚡ Speed: {len(genes)/file_parse_time:.0f} genes/s
💾 Saved to: {individual_csv_file.name}
🧬 Sequences included: {'Yes' if individual_results[accession]['has_sequences'] else 'No'}
📁 Progress: {i+1}/{total_files} files
🧬 Total genes: {total_genes_found}
                        """.strip()
                        progress_callback(overall_progress, status_msg)
                    
                    logging.info(f"✅ {accession}: Parsed {len(genes)} genes in {file_parse_time:.2f}s")
                else:
                    parsing_results[accession] = {
                        'status': 'no_genes',
                        'message': 'No genes found',
                        'parse_time': file_parse_time
                    }
                    logging.warning(f"⚠️ {accession}: No genes found")
            
            except Exception as e:
                parsing_results[accession] = {
                    'status': 'error',
                    'message': str(e),
                    'parse_time': 0
                }
                logging.error(f"❌ {accession}: Parsing error - {e}")
                
                if progress_callback:
                    progress_callback(overall_progress, f"❌ Error parsing {accession}: {str(e)}")
        
        # Also create a combined file for compatibility
        all_genes = []
        for acc, result in individual_results.items():
            for gene in result['genes']:
                gene['source_accession'] = acc
                gene['source_organism'] = download_results['organism']
                all_genes.append(gene)
        
        combined_csv_file = None
        if all_genes:
            combined_csv_file = output_folder / f"{download_results['organism'].replace(' ', '_')}_all_genes_combined.csv"
            df_combined = pd.DataFrame(all_genes)
            df_combined.to_csv(combined_csv_file, index=False)
        
        # Calculate final statistics
        avg_parse_time = total_parse_time / total_files if total_files > 0 else 0
        avg_genes_per_file = total_genes_found / total_files if total_files > 0 else 0
        
        if progress_callback:
            final_status = f"""
✅ Parsing completed!
📊 Total genes extracted: {total_genes_found}
📁 Files processed: {total_files}
⏱️ Total time: {total_parse_time:.1f}s
⚡ Average speed: {total_genes_found/total_parse_time:.0f} genes/s
📄 Average genes per file: {avg_genes_per_file:.0f}
💾 Individual files saved to: {output_folder}
            """.strip()
            progress_callback(100, final_status)
        
        return {
            'status': 'completed',
            'total_genes': total_genes_found,
            'individual_results': individual_results,
            'parsing_results': parsing_results,
            'combined_csv_file': str(combined_csv_file) if combined_csv_file else None,
            'output_folder': str(output_folder),
            'organism': download_results['organism'],
            'statistics': {
                'total_parse_time': total_parse_time,
                'average_parse_time': avg_parse_time,
                'average_genes_per_file': avg_genes_per_file,
                'parse_speed': total_genes_found / total_parse_time if total_parse_time > 0 else 0
            }
        }
    
    def get_parsing_statistics(self, parsing_results: Dict) -> Dict:
        """Get detailed parsing statistics"""
        stats = {
            'total_files': len(parsing_results),
            'successful': 0,
            'no_genes': 0,
            'errors': 0,
            'total_genes': 0,
            'total_parse_time': 0,
            'fastest_file': None,
            'slowest_file': None,
            'files_with_sequences': 0,
            'files_without_sequences': 0
        }
        
        fastest_time = float('inf')
        slowest_time = 0
        
        for accession, result in parsing_results.items():
            if result['status'] == 'success':
                stats['successful'] += 1
                stats['total_genes'] += result['gene_count']
                parse_time = result.get('parse_time', 0)
                stats['total_parse_time'] += parse_time
                
                # Check sequence status
                if result.get('has_sequences', False):
                    stats['files_with_sequences'] += 1
                else:
                    stats['files_without_sequences'] += 1
                
                # Track fastest and slowest
                if parse_time < fastest_time:
                    fastest_time = parse_time
                    stats['fastest_file'] = f"{accession} ({parse_time:.2f}s)"
                if parse_time > slowest_time:
                    slowest_time = parse_time
                    stats['slowest_file'] = f"{accession} ({parse_time:.2f}s)"
                    
            elif result['status'] == 'no_genes':
                stats['no_genes'] += 1
            else:
                stats['errors'] += 1
        
        return stats