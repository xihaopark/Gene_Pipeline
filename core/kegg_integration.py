# core/kegg_integration.py
import requests
import re
import time
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus

class KEGGIntegration:
    """KEGG数据库集成模块，支持批量基因组搜索和下载"""
    
    def __init__(self):
        self.base_url = "https://rest.kegg.jp"
        self.web_search_url = "https://www.genome.jp/dbget-bin/www_bfind_sub"
        self.rate_limit_delay = 0.5  # KEGG要求每秒不超过3次请求
        
    def search_organism_genomes(self, organism_name: str) -> List[Dict]:
        """搜索指定生物的基因组信息"""
        logging.info(f"Searching KEGG genomes for: {organism_name}")
        
        # 使用REST API搜索genome数据库
        genomes = self._search_genome_by_api(organism_name)
        
        # 去重并排序
        unique_genomes = self._deduplicate_genomes(genomes)
        
        logging.info(f"Found {len(unique_genomes)} unique genomes for {organism_name}")
        return unique_genomes
    
    def _search_genome_by_api(self, organism_name: str) -> List[Dict]:
        """使用KEGG REST API搜索基因组"""
        try:
            url = f"{self.base_url}/find/genome/{quote_plus(organism_name)}"
            response = requests.get(url)
            response.raise_for_status()
            
            genomes = []
            lines = response.text.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        genome_id = parts[0].strip()
                        description = parts[1].strip()
                        
                        # 提取T号
                        t_number = genome_id.replace('gn:', '')
                        
                        genomes.append({
                            'kegg_id': genome_id,
                            't_number': t_number,
                            'description': description,
                            'organism': self._extract_organism_name(description),
                            'source': 'kegg_api'
                        })
            
            time.sleep(self.rate_limit_delay)
            return genomes
            
        except Exception as e:
            logging.error(f"KEGG API search failed: {e}")
            return []
    
    def get_genome_details(self, t_number: str) -> Optional[Dict]:
        """获取指定T号基因组的详细信息，包括GenBank登录号"""
        try:
            url = f"{self.base_url}/get/gn:{t_number}"
            response = requests.get(url)
            response.raise_for_status()
            
            content = response.text
            logging.info(f"Processing KEGG genome entry for {t_number}")
            
            genome_info = {
                't_number': t_number,
                'kegg_id': f'gn:{t_number}',
                'genbank_accessions': [],
                'organism': '',
                'lineage': '',
                'data_source': '',
                'chromosomes': [],
                'organism_code': ''
            }
            
            # 首先尝试从KEGG条目中直接解析GenBank登录号
            genbank_accessions = self._parse_genbank_from_entry(content, genome_info)
            
            time.sleep(self.rate_limit_delay)
            
            # 如果直接解析没有找到，使用API方法
            if not genbank_accessions:
                logging.info(f"Direct parsing failed, trying API methods for {t_number}")
                genbank_accessions = self._get_genbank_via_api_methods(t_number, genome_info['organism_code'])
            
            genome_info['genbank_accessions'] = genbank_accessions
            
            # 为找到的登录号创建染色体条目
            for i, acc in enumerate(genbank_accessions):
                genome_info['chromosomes'].append({
                    'type': 'chromosome' if i == 0 else 'sequence',
                    'name': f'Sequence {i+1}',
                    'genbank_accession': acc,
                    'length': 0
                })
            
            if genome_info['genbank_accessions']:
                logging.info(f"Found {len(genome_info['genbank_accessions'])} GenBank accessions for {t_number}: {genome_info['genbank_accessions']}")
                return genome_info
            else:
                logging.warning(f"No GenBank accessions found for {t_number} using any method")
                return None
                
        except Exception as e:
            logging.error(f"Error getting genome details for {t_number}: {e}")
            return None
    
    def _parse_genbank_from_entry(self, content: str, genome_info: Dict) -> List[str]:
        """从KEGG基因组条目中直接解析GenBank登录号"""
        accessions = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # 解析基本信息
            if line.startswith('ENTRY'):
                match = re.search(r'ENTRY\s+(\S+)', line)
                if match:
                    genome_info['organism_code'] = match.group(1).strip()
            
            elif line.startswith('NAME'):
                genome_info['organism'] = line.replace('NAME', '').strip()
            
            elif line.startswith('TAXONOMY'):
                genome_info['lineage'] = line.replace('TAXONOMY', '').strip()
            
            elif line.startswith('DATA_SOURCE') or 'Data source' in line:
                genome_info['data_source'] = line.replace('DATA_SOURCE', '').strip()
                
                # 提取Assembly登录号
                assembly_match = re.search(r'Assembly:\s*([GC][CF]A_\d+\.\d+)', line, re.IGNORECASE)
                if assembly_match:
                    accessions.append(assembly_match.group(1))
            
            # 寻找Sequence行，格式如: "  Sequence GB:AM180355"
            elif 'Sequence' in line and 'GB:' in line:
                gb_match = re.search(r'GB:([A-Z]{1,4}\d{6,8}(?:\.\d+)?)', line, re.IGNORECASE)
                if gb_match:
                    accession = gb_match.group(1)
                    if accession not in accessions:
                        accessions.append(accession)
                        logging.info(f"Found GenBank accession in entry: {accession}")
            
            # 也检查其他可能包含GenBank ID的行
            elif any(keyword in line.upper() for keyword in ['CHROMOSOME', 'PLASMID', 'GENOME', 'SEQUENCE']):
                # 寻找各种GenBank格式
                patterns = [
                    r'\b([A-Z]{2}\d{6}(?:\.\d+)?)\b',  # 标准格式: AM180355
                    r'\b(NC_\d{6}(?:\.\d+)?)\b',      # RefSeq: NC_000913.3
                    r'\b(NZ_[A-Z]{2,4}\d{6,8}(?:\.\d+)?)\b',  # NZ_格式
                    r'\b(CP\d{6}(?:\.\d+)?)\b',       # CP格式
                    r'\b(AP\d{6}(?:\.\d+)?)\b',       # AP格式
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, line)
                    for match in matches:
                        if match not in accessions and not match.startswith('T0'):  # 避免T号
                            accessions.append(match)
                            logging.info(f"Found potential GenBank accession: {match}")
        
        return accessions
    
    def _get_genbank_via_api_methods(self, t_number: str, organism_code: str) -> List[str]:
        """使用KEGG API方法动态获取GenBank登录号"""
        genbank_accessions = []
        
        # 方法1: 通过link操作从genome到genes
        try:
            logging.info(f"Trying link operation for {t_number}")
            url = f"{self.base_url}/link/genes/gn:{t_number}"
            response = requests.get(url)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                content = response.text.strip()
                if content:
                    # 从link结果中提取基因ID
                    gene_ids = []
                    for line in content.split('\n'):
                        if line.strip():
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                gene_id = parts[1].strip()
                                gene_ids.append(gene_id)
                    
                    # 从基因中提取基因组信息
                    if gene_ids:
                        accessions = self._extract_genbank_from_genes(gene_ids[:3])  # 只检查前3个
                        genbank_accessions.extend(accessions)
        except Exception as e:
            logging.error(f"Link operation failed: {e}")
        
        # 方法2: 如果有organism code，尝试直接获取
        if not genbank_accessions and organism_code and organism_code != t_number:
            try:
                logging.info(f"Trying organism code method for {organism_code}")
                accessions = self._get_genbank_from_organism_code(organism_code)
                genbank_accessions.extend(accessions)
            except Exception as e:
                logging.error(f"Organism code method failed: {e}")
        
        # 去重
        unique_accessions = list(set(genbank_accessions))
        
        if unique_accessions:
            logging.info(f"API methods found GenBank accessions: {unique_accessions}")
        else:
            logging.warning(f"All API methods failed for {t_number}")
        
        return unique_accessions
    
    def _extract_genbank_from_genes(self, gene_ids: List[str]) -> List[str]:
        """从基因ID中提取GenBank序列信息"""
        accessions = []
        
        try:
            # 获取基因详情
            for gene_id in gene_ids[:3]:  # 只检查前3个基因
                url = f"{self.base_url}/get/{gene_id}"
                response = requests.get(url)
                time.sleep(self.rate_limit_delay)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # 在基因条目中寻找染色体/序列信息
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('POSITION') or 'chromosome' in line.lower():
                            # 提取可能的GenBank登录号
                            potential_accessions = re.findall(r'([A-Z]{2,4}[0-9]{6,8}(?:\.[0-9]+)?)', line)
                            accessions.extend(potential_accessions)
                
                # 限制API调用频率
                if len(accessions) > 0:
                    break  # 找到一些就停止
                    
        except Exception as e:
            logging.error(f"Error extracting GenBank from genes: {e}")
        
        return list(set(accessions))
    
    def _get_genbank_from_organism_code(self, organism_code: str) -> List[str]:
        """通过organism code获取GenBank信息"""
        accessions = []
        
        try:
            # 获取organism的基因列表
            url = f"{self.base_url}/list/{organism_code}"
            response = requests.get(url)
            time.sleep(self.rate_limit_delay)
            
            if response.status_code == 200:
                content = response.text
                lines = content.split('\n')[:10]  # 只看前10行
                
                for line in lines:
                    # 从基因条目描述中提取可能的GenBank信息
                    potential_accessions = re.findall(r'([A-Z]{2,4}[0-9]{6,8}(?:\.[0-9]+)?)', line)
                    accessions.extend(potential_accessions)
                    
        except Exception as e:
            logging.error(f"Error getting GenBank from organism code: {e}")
        
        return list(set(accessions))
    
    def batch_get_genbank_accessions(self, t_numbers: List[str]) -> Dict[str, Dict]:
        """批量获取多个T号对应的GenBank登录号"""
        logging.info(f"Batch processing {len(t_numbers)} KEGG genomes")
        
        results = {}
        failed_count = 0
        
        for i, t_number in enumerate(t_numbers):
            logging.info(f"Processing {i+1}/{len(t_numbers)}: {t_number}")
            
            genome_info = self.get_genome_details(t_number)
            if genome_info and genome_info['genbank_accessions']:
                results[t_number] = {
                    'genbank_accessions': genome_info['genbank_accessions'],
                    'organism': genome_info['organism'],
                    'chromosomes': genome_info['chromosomes']
                }
            else:
                failed_count += 1
                logging.warning(f"Failed to get GenBank accessions for {t_number}")
            
            # 进度报告
            if (i + 1) % 10 == 0:
                logging.info(f"Progress: {i+1}/{len(t_numbers)} completed, {failed_count} failed")
        
        logging.info(f"Batch processing completed: {len(results)}/{len(t_numbers)} successful")
        return results
    
    def _extract_organism_name(self, description: str) -> str:
        """从描述中提取生物名称"""
        # 移除常见的后缀和括号内容
        cleaned = re.sub(r'\([^)]*\)', '', description)
        cleaned = re.sub(r'\s*strain\s+\S+.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*isolate\s+\S+.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*serovar\s+\S+.*', '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip()
    
    def _deduplicate_genomes(self, genomes: List[Dict]) -> List[Dict]:
        """去除重复的基因组条目"""
        seen = set()
        unique_genomes = []
        
        for genome in genomes:
            key = genome['t_number']
            if key not in seen:
                seen.add(key)
                unique_genomes.append(genome)
        
        # 按T号排序
        unique_genomes.sort(key=lambda x: x['t_number'])
        return unique_genomes
    
    def search_and_extract_all(self, organism_name: str) -> Tuple[List[Dict], Dict[str, Dict]]:
        """一站式搜索：从生物名称到GenBank登录号"""
        # 第一步：搜索基因组
        genomes = self.search_organism_genomes(organism_name)
        
        if not genomes:
            logging.warning(f"No genomes found for {organism_name}")
            return [], {}
        
        # 第二步：批量获取GenBank登录号
        t_numbers = [g['t_number'] for g in genomes]
        genbank_mapping = self.batch_get_genbank_accessions(t_numbers)
        
        return genomes, genbank_mapping