# core/sequence_analyzer.py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import streamlit as st
import logging
from io import StringIO

class SequenceAnalyzer:
    """Advanced sequence analysis with clustering and visualization"""
    
    def __init__(self):
        self.kmer_size = 8
        self.clustering_method = 'kmeans'
    
    def run_complete_mapping_analysis(self, query_file: str, target_files: List[str], 
                                     file_data: Dict[str, pd.DataFrame], threshold: float,
                                     use_clustering: bool = True, n_clusters: int = 20,
                                     show_visualizations: bool = True) -> Dict:
        """Complete mapping analysis workflow with optional clustering"""
        
        if use_clustering:
            return self.run_clustered_mapping(query_file, target_files, file_data, 
                                            threshold, n_clusters, show_visualizations)
        else:
            return self.run_simple_mapping(query_file, target_files, file_data, threshold)
    
    def run_simple_mapping(self, query_file: str, target_files: List[str],
                          file_data: Dict[str, pd.DataFrame], threshold: float) -> Dict:
        """Run simple pairwise mapping without clustering"""
        
        # Get query sequences
        query_df = file_data[query_file]
        query_df = query_df.drop_duplicates(subset=['Locus Tag'], keep='first')
        query_df_filtered = query_df[query_df['NT Seq'].notna() & (query_df['NT Seq'] != '')]
        
        if query_df_filtered.empty:
            return {'status': 'error', 'message': 'No sequences found in query file'}
        
        # Pre-process query sequences
        query_data = []
        for _, row in query_df_filtered.iterrows():
            query_data.append({
                'locus_tag': row['Locus Tag'],
                'gene': row.get('Gene', 'N/A'),
                'sequence': row['NT Seq']
            })
        
        all_results = []
        total_comparisons = 0
        
        for target_file in target_files:
            # Get target data
            target_df = file_data[target_file]
            target_df = target_df.drop_duplicates(subset=['Locus Tag'], keep='first')
            
            # Create target dictionary
            target_df_filtered = target_df[target_df['NT Seq'].notna() & (target_df['NT Seq'] != '')]
            target_dict = {}
            for _, row in target_df_filtered.iterrows():
                target_dict[row['Locus Tag']] = {
                    'NT Seq': row['NT Seq'],
                    'Gene': row.get('Gene', 'N/A')
                }
            
            # Run comparisons
            for query_info in query_data:
                best_match, best_score, best_details = self.find_best_match_optimized(
                    query_info['sequence'], target_dict
                )
                
                total_comparisons += len(target_dict)
                
                if best_score >= threshold:
                    all_results.append({
                        'Query File': query_file,
                        'Target File': target_file,
                        'Query Locus Tag': query_info['locus_tag'],
                        'Query Gene': query_info['gene'],
                        'Best Match Locus Tag': best_match,
                        'Best Match Gene': best_details.get('Gene', 'N/A') if best_details else 'N/A',
                        'Match Score': round(best_score, 3)
                    })
        
        return {
            'status': 'success',
            'matches': all_results,
            'total_comparisons': total_comparisons,
            'query_count': len(query_data),
            'results_df': pd.DataFrame(all_results) if all_results else pd.DataFrame()
        }
    
    def run_clustered_mapping(self, query_file: str, target_files: List[str],
                             file_data: Dict[str, pd.DataFrame], threshold: float,
                             n_clusters: int, show_visualizations: bool) -> Dict:
        """Run clustering-based mapping analysis"""
        
        # Prepare all sequences
        all_sequences = []
        all_labels = []
        sequence_to_info = {}
        
        # Add query sequences
        query_df = file_data[query_file]
        query_df = query_df.drop_duplicates(subset=['Locus Tag'], keep='first')
        query_df_filtered = query_df[query_df['NT Seq'].notna() & (query_df['NT Seq'] != '')]
        
        for _, row in query_df_filtered.iterrows():
            seq = row['NT Seq']
            label = f"{query_file}_{row['Locus Tag']}"
            all_sequences.append(seq)
            all_labels.append(label)
            sequence_to_info[label] = {
                'file': query_file,
                'locus_tag': row['Locus Tag'],
                'gene': row.get('Gene', 'N/A'),
                'type': 'query'
            }
        
        # Add target sequences
        for target_file in target_files:
            target_df = file_data[target_file]
            target_df = target_df.drop_duplicates(subset=['Locus Tag'], keep='first')
            target_df_filtered = target_df[target_df['NT Seq'].notna() & (target_df['NT Seq'] != '')]
            
            for _, row in target_df_filtered.iterrows():
                seq = row['NT Seq']
                label = f"{target_file}_{row['Locus Tag']}"
                all_sequences.append(seq)
                all_labels.append(label)
                sequence_to_info[label] = {
                    'file': target_file,
                    'locus_tag': row['Locus Tag'],
                    'gene': row.get('Gene', 'N/A'),
                    'type': 'target'
                }
        
        # Cluster sequences
        cluster_result = self.cluster_sequences(all_sequences, all_labels, n_clusters)
        
        # Match within clusters
        matches = []
        total_comparisons = 0
        
        for cluster_id, members in cluster_result['clusters'].items():
            # Separate query and target sequences
            cluster_queries = [m for m in members if sequence_to_info[m['label']]['type'] == 'query']
            cluster_targets = [m for m in members if sequence_to_info[m['label']]['type'] == 'target']
            
            # Count comparisons
            cluster_comparisons = len(cluster_queries) * len(cluster_targets)
            total_comparisons += cluster_comparisons
            
            # Match within cluster
            for query in cluster_queries:
                query_seq = query['sequence']
                query_info = sequence_to_info[query['label']]
                
                for target in cluster_targets:
                    target_seq = target['sequence']
                    target_info = sequence_to_info[target['label']]
                    
                    # Calculate similarity
                    similarity = self.quick_similarity(query_seq, target_seq)
                    
                    if similarity >= threshold:
                        matches.append({
                            'Query File': query_info['file'],
                            'Target File': target_info['file'],
                            'Query Locus Tag': query_info['locus_tag'],
                            'Query Gene': query_info['gene'],
                            'Best Match Locus Tag': target_info['locus_tag'],
                            'Best Match Gene': target_info['gene'],
                            'Match Score': round(similarity, 3),
                            'Cluster': cluster_id
                        })
        
        # Calculate saved comparisons
        total_possible = len([s for s in sequence_to_info.values() if s['type'] == 'query']) * \
                        len([s for s in sequence_to_info.values() if s['type'] == 'target'])
        saved_comparisons = total_possible - total_comparisons
        
        result = {
            'status': 'success',
            'matches': matches,
            'total_comparisons': total_comparisons,
            'saved_comparisons': saved_comparisons,
            'cluster_result': cluster_result,
            'sequence_info': sequence_to_info,
            'results_df': pd.DataFrame(matches) if matches else pd.DataFrame(),
            'show_visualizations': show_visualizations,
            'all_files': [query_file] + target_files
        }
        
        return result
    
    def find_best_match_optimized(self, query_seq: str, db_seqs: Dict) -> tuple:
        """Optimized sequence matching with k-mer prescreening"""
        best_match = None
        best_score = 0
        best_details = {}
        
        if not isinstance(query_seq, str) or not query_seq:
            return None, 0, {}
        
        query_len = len(query_seq)
        
        # Create k-mers for query sequence
        k = 8
        query_kmers = set()
        if query_len >= k:
            for i in range(query_len - k + 1):
                query_kmers.add(query_seq[i:i+k])
        
        # Screen candidates
        candidates = []
        
        for tag, details in db_seqs.items():
            if not isinstance(details, dict):
                continue
            
            db_seq = details.get('NT Seq', '')
            if not isinstance(db_seq, str) or not db_seq or pd.isna(db_seq):
                continue
            
            db_len = len(db_seq)
            
            # Quick length check
            length_ratio = min(query_len, db_len) / max(query_len, db_len)
            if length_ratio < 0.5:
                continue
            
            # K-mer screening
            if query_len >= k and db_len >= k:
                db_kmers = set()
                for i in range(min(db_len - k + 1, query_len * 2)):
                    db_kmers.add(db_seq[i:i+k])
                
                intersection = len(query_kmers & db_kmers)
                union = len(query_kmers | db_kmers)
                
                if union > 0:
                    kmer_similarity = intersection / union
                    if kmer_similarity > 0.3:
                        candidates.append((tag, details, db_seq, kmer_similarity))
            else:
                candidates.append((tag, details, db_seq, 1.0))
        
        # Detailed comparison for candidates
        for tag, details, db_seq, kmer_sim in sorted(candidates, key=lambda x: x[3], reverse=True):
            try:
                if query_seq == db_seq:
                    best_score = 1.0
                    best_match = tag
                    best_details = details
                    continue
                
                # Sliding window alignment
                min_len = min(query_len, len(db_seq))
                if min_len > 0:
                    max_score = 0
                    shift_range = min(10, abs(query_len - len(db_seq)) + 1)
                    
                    for shift in range(-shift_range, shift_range + 1):
                        matches = 0
                        comparisons = 0
                        
                        for i in range(min_len):
                            q_idx = i
                            d_idx = i + shift
                            
                            if 0 <= q_idx < query_len and 0 <= d_idx < len(db_seq):
                                comparisons += 1
                                if query_seq[q_idx] == db_seq[d_idx]:
                                    matches += 1
                        
                        if comparisons > 0:
                            score = matches / comparisons
                            max_score = max(max_score, score)
                    
                    # Weight by length similarity
                    length_similarity = min_len / max(query_len, len(db_seq))
                    final_score = max_score * (0.8 + 0.2 * length_similarity)
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_match = tag
                        best_details = details
                        
            except Exception:
                continue
        
        return best_match, best_score, best_details
    
    def extract_features(self, sequences: List[str], feature_type: str = 'kmer') -> np.ndarray:
        """Extract numerical features from sequences for clustering"""
        
        if feature_type == 'kmer':
            return self._extract_kmer_features(sequences)
        elif feature_type == 'composition':
            return self._extract_composition_features(sequences)
        else:
            return self._extract_combined_features(sequences)
    
    def _extract_kmer_features(self, sequences: List[str], k: int = 4) -> np.ndarray:
        """Extract k-mer frequency features"""
        from itertools import product
        
        # Generate all possible k-mers
        bases = ['A', 'T', 'G', 'C']
        all_kmers = [''.join(p) for p in product(bases, repeat=k)]
        kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
        
        # Create feature matrix
        features = np.zeros((len(sequences), len(all_kmers)))
        
        for i, seq in enumerate(sequences):
            seq = seq.upper()
            seq_len = len(seq)
            
            # Count k-mers
            kmer_counts = defaultdict(int)
            for j in range(len(seq) - k + 1):
                kmer = seq[j:j+k]
                if kmer in kmer_to_idx:
                    kmer_counts[kmer] += 1
            
            # Normalize by sequence length
            for kmer, count in kmer_counts.items():
                features[i, kmer_to_idx[kmer]] = count / (seq_len - k + 1) if seq_len >= k else 0
        
        return features
    
    def _extract_composition_features(self, sequences: List[str]) -> np.ndarray:
        """Extract nucleotide composition features"""
        features = []
        
        for seq in sequences:
            seq = seq.upper()
            length = len(seq)
            
            if length == 0:
                features.append([0, 0, 0, 0, 0])
                continue
            
            # Calculate composition
            a_count = seq.count('A') / length
            t_count = seq.count('T') / length
            g_count = seq.count('G') / length
            c_count = seq.count('C') / length
            gc_content = (g_count + c_count)
            
            features.append([a_count, t_count, g_count, c_count, gc_content])
        
        return np.array(features)
    
    def _extract_combined_features(self, sequences: List[str]) -> np.ndarray:
        """Combine multiple feature types"""
        kmer_features = self._extract_kmer_features(sequences, k=3)
        comp_features = self._extract_composition_features(sequences)
        
        # Normalize each feature set
        scaler = StandardScaler()
        kmer_features = scaler.fit_transform(kmer_features)
        comp_features = scaler.fit_transform(comp_features)
        
        # Combine features
        return np.hstack([kmer_features, comp_features])
    
    def cluster_sequences(self, sequences: List[str], labels: List[str], 
                         n_clusters: int = 10, method: str = 'kmeans') -> Dict:
        """Cluster sequences and return cluster assignments"""
        
        # Extract features
        features = self.extract_features(sequences, 'combined')
        
        # Apply clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(features)
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.3, min_samples=5)
            cluster_labels = clusterer.fit_predict(features)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        # Organize results
        clusters = defaultdict(list)
        for i, (seq, label, cluster) in enumerate(zip(sequences, labels, cluster_labels)):
            clusters[cluster].append({
                'index': i,
                'sequence': seq,
                'label': label,
                'file': label.split('_')[0] if '_' in label else 'unknown'
            })
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id, members in clusters.items():
            files_in_cluster = defaultdict(int)
            for member in members:
                files_in_cluster[member['file']] += 1
            
            cluster_stats[cluster_id] = {
                'size': len(members),
                'file_distribution': dict(files_in_cluster),
                'diversity': len(files_in_cluster)
            }
        
        return {
            'clusters': dict(clusters),
            'features': features,
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'cluster_stats': cluster_stats
        }
    
    def fast_cluster_based_matching(self, query_sequences: Dict[str, str], 
                                   target_sequences: Dict[str, str],
                                   n_clusters: int = 20,
                                   similarity_threshold: float = 0.7) -> List[Dict]:
        """Fast matching using clustering to reduce comparisons"""
        
        # Combine all sequences for clustering
        all_sequences = []
        all_labels = []
        query_indices = []
        
        for label, seq in query_sequences.items():
            all_sequences.append(seq)
            all_labels.append(f"query_{label}")
            query_indices.append(len(all_sequences) - 1)
        
        target_start_idx = len(all_sequences)
        for label, seq in target_sequences.items():
            all_sequences.append(seq)
            all_labels.append(f"target_{label}")
        
        # Cluster all sequences
        cluster_result = self.cluster_sequences(all_sequences, all_labels, n_clusters)
        
        # Match within clusters only
        matches = []
        
        for cluster_id, members in cluster_result['clusters'].items():
            # Separate query and target sequences in this cluster
            cluster_queries = [m for m in members if m['label'].startswith('query_')]
            cluster_targets = [m for m in members if m['label'].startswith('target_')]
            
            # Compare within cluster
            for query in cluster_queries:
                query_seq = query['sequence']
                query_label = query['label'].replace('query_', '')
                
                for target in cluster_targets:
                    target_seq = target['sequence']
                    target_label = target['label'].replace('target_', '')
                    
                    # Quick similarity check
                    similarity = self.quick_similarity(query_seq, target_seq)
                    
                    if similarity >= similarity_threshold:
                        matches.append({
                            'query': query_label,
                            'target': target_label,
                            'similarity': similarity,
                            'cluster': cluster_id
                        })
        
        return matches
    
    def quick_similarity(self, seq1: str, seq2: str) -> float:
        """Quick sequence similarity calculation"""
        if not seq1 or not seq2:
            return 0.0
        
        # Length similarity check
        len_ratio = min(len(seq1), len(seq2)) / max(len(seq1), len(seq2))
        if len_ratio < 0.5:
            return 0.0
        
        # Use k-mer Jaccard similarity for speed
        k = 8
        if len(seq1) < k or len(seq2) < k:
            # For short sequences, use simple comparison
            matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
            return matches / max(len(seq1), len(seq2))
        
        # K-mer similarity
        kmers1 = set(seq1[i:i+k] for i in range(len(seq1) - k + 1))
        kmers2 = set(seq2[i:i+k] for i in range(len(seq2) - k + 1))
        
        intersection = len(kmers1 & kmers2)
        union = len(kmers1 | kmers2)
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        
        # Weight by length similarity
        return jaccard_sim * (0.7 + 0.3 * len_ratio)
    
    def create_cluster_visualization(self, cluster_result: Dict, method: str = 'pca') -> go.Figure:
        """Create cluster visualization using PCA or t-SNE"""
        
        features = cluster_result['features']
        labels = cluster_result['cluster_labels']
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            coords = reducer.fit_transform(features)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            coords = reducer.fit_transform(features)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot each cluster
        unique_clusters = sorted(set(labels))
        colors = px.colors.qualitative.Set3
        
        for i, cluster in enumerate(unique_clusters):
            cluster_mask = labels == cluster
            cluster_coords = coords[cluster_mask]
            
            # Get file information for hover text
            hover_texts = []
            for j, member in enumerate(cluster_result['clusters'][cluster]):
                hover_texts.append(f"File: {member['file']}<br>Label: {member['label']}")
            
            fig.add_trace(go.Scatter(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(
                    size=8,
                    color=colors[i % len(colors)],
                    line=dict(width=1, color='white')
                ),
                text=hover_texts,
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title='Sequence Cluster Visualization',
            xaxis_title='Component 1',
            yaxis_title='Component 2',
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def create_sankey_diagram(self, matches: List[Dict], files: List[str]) -> go.Figure:
        """Create Sankey diagram showing gene flow between files"""
        
        # Prepare data for Sankey
        source_indices = []
        target_indices = []
        values = []
        
        # Create file to index mapping
        file_to_idx = {file: i for i, file in enumerate(files)}
        
        # Count matches between files
        match_counts = defaultdict(lambda: defaultdict(int))
        for match in matches:
            source_file = match.get('Query File', match.get('query', '')).split('_')[0]
            target_file = match.get('Target File', match.get('target', '')).split('_')[0]
            if source_file in files and target_file in files:
                match_counts[source_file][target_file] += 1
        
        # Build Sankey data
        for source_file, targets in match_counts.items():
            for target_file, count in targets.items():
                if source_file != target_file:  # Avoid self-loops
                    source_indices.append(file_to_idx[source_file])
                    target_indices.append(file_to_idx[target_file] + len(files))
                    values.append(count)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=files + [f"{f} (matched)" for f in files],
                color="lightblue"
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )])
        
        fig.update_layout(
            title="Gene Match Flow Between Files",
            font_size=10,
            height=600
        )
        
        return fig
    
    def create_heatmap(self, matches: List[Dict], query_files: List[str], 
                      target_files: List[str]) -> go.Figure:
        """Create heatmap of match counts between files"""
        
        # Create matrix
        matrix = np.zeros((len(query_files), len(target_files)))
        
        # Count matches
        for match in matches:
            query_file = match.get('Query File', '').split('_')[0]
            target_file = match.get('Target File', '').split('_')[0]
            
            if query_file in query_files and target_file in target_files:
                i = query_files.index(query_file)
                j = target_files.index(target_file)
                matrix[i, j] += 1
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=target_files,
            y=query_files,
            colorscale='Viridis',
            text=matrix.astype(int),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Gene Match Heatmap',
            xaxis_title='Target Files',
            yaxis_title='Query Files',
            height=600
        )
        
        return fig
    
    def create_enrichment_plot(self, cluster_result: Dict) -> go.Figure:
        """Create enrichment plot showing file distribution in clusters"""
        
        # Calculate enrichment scores
        cluster_stats = cluster_result['cluster_stats']
        
        # Prepare data for grouped bar chart
        clusters = []
        files = set()
        
        for cluster_id, stats in cluster_stats.items():
            for file in stats['file_distribution']:
                files.add(file)
        
        files = sorted(list(files))
        
        # Create data for each file
        data = []
        for file in files:
            counts = []
            for cluster_id in sorted(cluster_stats.keys()):
                count = cluster_stats[cluster_id]['file_distribution'].get(file, 0)
                counts.append(count)
            
            data.append(go.Bar(
                name=file,
                x=[f'Cluster {i}' for i in sorted(cluster_stats.keys())],
                y=counts
            ))
        
        fig = go.Figure(data=data)
        
        fig.update_layout(
            title='Gene Distribution Across Clusters',
            xaxis_title='Clusters',
            yaxis_title='Number of Genes',
            barmode='stack',
            height=600
        )
        
        return fig