# core/batch_processor.py
import logging
import time
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from .genbank_processor import GenBankProcessor
from .kegg_integration import KEGGIntegration

class BatchGenomeProcessor:
    """批量基因组处理器"""
    
    def __init__(self, max_workers: int = 3):
        self.genbank_processor = GenBankProcessor()
        self.kegg_integration = KEGGIntegration()
        self.max_workers = max_workers
        
    def process_organism_batch(self, organism_name: str, max_genomes: Optional[int] = None) -> Dict:
        """
        批量处理指定生物的所有基因组
        
        Args:
            organism_name: 生物名称，如 "Clostridioides difficile"
            max_genomes: 最大处理基因组数量，None表示处理所有
            
        Returns:
            Dict: 包含处理结果的字典
        """
        logging.info(f"Starting batch processing for organism: {organism_name}")
        
        # 第一步：从KEGG搜索基因组
        genomes, genbank_mapping = self.kegg_integration.search_and_extract_all(organism_name)
        
        if not genomes:
            return {
                'status': 'failed',
                'message': f'No genomes found for {organism_name}',
                'results': []
            }
        
        # 限制处理数量
        if max_genomes and len(genomes) > max_genomes:
            genomes = genomes[:max_genomes]
            logging.info(f"Limited processing to first {max_genomes} genomes")
        
        # 第二步：收集所有GenBank登录号
        all_accessions = []
        genome_accession_map = {}
        
        for genome in genomes:
            t_number = genome['t_number']
            if t_number in genbank_mapping:
                accessions = genbank_mapping[t_number]['genbank_accessions']
                all_accessions.extend(accessions)
                genome_accession_map[t_number] = accessions
        
        logging.info(f"Found {len(all_accessions)} GenBank accessions to process")
        
        # 第三步：批量处理GenBank记录
        results = self._batch_process_genbank_accessions(all_accessions, organism_name)
        
        # 第四步：组织结果
        return self._organize_batch_results(genomes, genbank_mapping, results, organism_name)
    
    def _batch_process_genbank_accessions(self, accessions: List[str], organism_name: str) -> Dict[str, Dict]:
        """批量处理GenBank登录号"""
        results = {}
        
        logging.info(f"Processing {len(accessions)} GenBank accessions")
        
        # 串行处理以避免NCBI API限制
        for i, accession in enumerate(accessions):
            logging.info(f"Processing {i+1}/{len(accessions)}: {accession}")
            
            try:
                # 下载GenBank数据
                genbank_data = self.genbank_processor.fetch_genbank(accession, save_to_cache=True)
                
                if genbank_data:
                    # 解析基因
                    genes = self.genbank_processor.parse_genes(genbank_data)
                    
                    if genes:
                        # 保存到缓存
                        self.genbank_processor.save_genes_to_cache(genes, accession)
                        
                        results[accession] = {
                            'status': 'success',
                            'gene_count': len(genes),
                            'genes': genes,
                            'genbank_size': len(genbank_data),
                            'organism': organism_name
                        }
                        
                        logging.info(f"✅ {accession}: {len(genes)} genes extracted")
                    else:
                        results[accession] = {
                            'status': 'no_genes',
                            'message': 'No genes found in GenBank record',
                            'organism': organism_name
                        }
                        logging.warning(f"⚠️ {accession}: No genes found")
                else:
                    results[accession] = {
                        'status': 'download_failed',
                        'message': 'Failed to download GenBank data',
                        'organism': organism_name
                    }
                    logging.error(f"❌ {accession}: Download failed")
                    
            except Exception as e:
                results[accession] = {
                    'status': 'error',
                    'message': str(e),
                    'organism': organism_name
                }
                logging.error(f"❌ {accession}: Error - {e}")
            
            # 短暂延迟以避免API限制
            time.sleep(0.5)
        
        return results
    
    def _organize_batch_results(self, genomes: List[Dict], genbank_mapping: Dict, 
                               processing_results: Dict, organism_name: str) -> Dict:
        """组织批量处理结果"""
        
        organized_results = {
            'organism': organism_name,
            'total_genomes_found': len(genomes),
            'total_accessions_processed': len(processing_results),
            'successful_extractions': 0,
            'total_genes_extracted': 0,
            'genomes': [],
            'summary': {},
            'all_genes': []  # 合并所有基因
        }
        
        # 统计和组织每个基因组的结果
        for genome in genomes:
            t_number = genome['t_number']
            
            genome_result = {
                'kegg_info': genome,
                'accessions': [],
                'gene_extractions': [],
                'total_genes': 0,
                'status': 'not_processed'
            }
            
            if t_number in genbank_mapping:
                accessions = genbank_mapping[t_number]['genbank_accessions']
                genome_result['accessions'] = accessions
                
                successful_extractions = 0
                for accession in accessions:
                    if accession in processing_results:
                        result = processing_results[accession]
                        genome_result['gene_extractions'].append({
                            'accession': accession,
                            'status': result['status'],
                            'gene_count': result.get('gene_count', 0),
                            'message': result.get('message', '')
                        })
                        
                        if result['status'] == 'success':
                            successful_extractions += 1
                            genome_result['total_genes'] += result['gene_count']
                            organized_results['total_genes_extracted'] += result['gene_count']
                            
                            # 添加基因到总列表，标记来源
                            for gene in result['genes']:
                                gene_with_source = gene.copy()
                                gene_with_source['source_accession'] = accession
                                gene_with_source['kegg_t_number'] = t_number
                                gene_with_source['source_organism'] = organism_name
                                organized_results['all_genes'].append(gene_with_source)
                
                if successful_extractions > 0:
                    genome_result['status'] = 'success'
                    organized_results['successful_extractions'] += 1
                elif len(genome_result['gene_extractions']) > 0:
                    genome_result['status'] = 'partial_failure'
                else:
                    genome_result['status'] = 'failed'
            
            organized_results['genomes'].append(genome_result)
        
        # 生成汇总统计
        organized_results['summary'] = self._generate_summary_stats(organized_results)
        
        return organized_results
    
    def _generate_summary_stats(self, results: Dict) -> Dict:
        """生成汇总统计信息"""
        summary = {
            'organism': results['organism'],
            'genomes_found': results['total_genomes_found'],
            'genomes_with_genes': results['successful_extractions'],
            'total_genes': results['total_genes_extracted'],
            'accessions_processed': results['total_accessions_processed']
        }
        
        # 计算成功率
        if results['total_genomes_found'] > 0:
            summary['success_rate'] = f"{(results['successful_extractions'] / results['total_genomes_found'] * 100):.1f}%"
        
        # 基因功能统计
        if results['all_genes']:
            genes = results['all_genes']
            summary['genes_with_protein_id'] = sum(1 for g in genes if g.get('protein_id'))
            summary['genes_with_ec_number'] = sum(1 for g in genes if g.get('ec_number'))
            summary['cds_features'] = sum(1 for g in genes if g.get('feature_type') == 'CDS')
        
        return summary
    
    def export_batch_results_to_csv(self, results: Dict, include_sequences: bool = False) -> str:
        """
        将批量处理结果导出为CSV
        
        Args:
            results: 批量处理结果
            include_sequences: 是否包含序列信息
            
        Returns:
            str: CSV内容
        """
        if not results['all_genes']:
            return "No genes to export"
        
        # 创建DataFrame
        df = pd.DataFrame(results['all_genes'])
        
        # 重新排列列顺序
        base_columns = [
            'source_organism', 'kegg_t_number', 'source_accession',
            'locus_tag', 'gene', 'product', 'start', 'end', 'strand',
            'length', 'protein_id', 'ec_number', 'feature_type'
        ]
        
        if include_sequences:
            base_columns.extend(['sequence', 'translation'])
        
        # 只包含存在的列
        available_columns = [col for col in base_columns if col in df.columns]
        df = df[available_columns]
        
        return df.to_csv(index=False)
    
    def get_processing_progress(self) -> Dict:
        """获取当前处理进度（可用于实时更新UI）"""
        # 这里可以添加进度跟踪逻辑
        # 目前返回基本信息
        return {
            'status': 'idle',  # idle, processing, completed, error
            'current_step': '',
            'progress_percentage': 0,
            'current_genome': '',
            'current_accession': ''
        }