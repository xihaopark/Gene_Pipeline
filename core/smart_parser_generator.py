# core/smart_parser_generator.py
import re
import time
import logging
from typing import Dict, List, Optional, Tuple
import streamlit as st

# Import API key
try:
    from api import ANTHROPIC_API_KEY
except ImportError:
    ANTHROPIC_API_KEY = None
    logging.warning("api.py not found. Please create api.py with ANTHROPIC_API_KEY")

# Try to import anthropic
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic library not installed. Claude parsing option will be disabled.")

class SmartParserGenerator:
    """Generate optimized parsers for GenBank files using Claude API"""
    
    def __init__(self):
        """Initialize the parser generator"""
        self.client = None
        self._parser_cache = {}
        self._initialized = False  # Prevent multiple initializations
        
        # Auto-initialize with API key from api.py
        if not self._initialized and ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY:
            try:
                self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
                logging.info("Claude API client initialized with api.py key")
                self._initialized = True
            except Exception as e:
                logging.error(f"Failed to initialize Claude client: {e}")
        
    def init_claude_client(self, api_key: str) -> bool:
        """Initialize Claude client with API key"""
        if self._initialized and self.client:
            return True  # Already initialized
            
        if not ANTHROPIC_AVAILABLE:
            return False
            
        if api_key:
            try:
                self.client = Anthropic(api_key=api_key)
                logging.info("Claude API client initialized successfully")
                self._initialized = True
                return True
            except Exception as e:
                logging.error(f"Failed to initialize Claude client: {e}")
                return False
        return False
    
    def analyze_genbank_structure(self, genbank_content: str) -> Dict:
        """Analyze GenBank file structure and extract MORE samples for better accuracy"""
        lines = genbank_content.split('\n')
        
        structure = {
            'header_sample': '',
            'feature_samples': [],
            'sequence_sample': '',
            'file_size': len(genbank_content),
            'total_lines': len(lines),
            'feature_variety': {}  # Track different types of features
        }
        
        # Find key sections
        features_start = -1
        origin_start = -1
        
        for i, line in enumerate(lines):
            if line.startswith('FEATURES'):
                features_start = i
            elif line.startswith('ORIGIN'):
                origin_start = i
                break
        
        # Extract header (first 50 lines for more context)
        structure['header_sample'] = '\n'.join(lines[:min(50, len(lines))])
        
        # Extract MORE feature samples - aim for 20 diverse samples
        if features_start > 0:
            features_end = origin_start if origin_start > 0 else len(lines)
            feature_lines = lines[features_start:features_end]
            
            # Collect different types of features
            feature_blocks = {}
            current_feature = []
            current_type = None
            
            for line in feature_lines:
                # Check if it's a new feature
                if re.match(r'^\s{5}(\w+)\s+', line):
                    # Save previous feature
                    if current_feature and current_type:
                        if current_type not in feature_blocks:
                            feature_blocks[current_type] = []
                        feature_blocks[current_type].append('\n'.join(current_feature))
                    
                    # Start new feature
                    match = re.match(r'^\s{5}(\w+)\s+', line)
                    current_type = match.group(1)
                    current_feature = [line]
                elif current_feature and line.startswith('                     '):
                    # Continue current feature
                    current_feature.append(line)
            
            # Save last feature
            if current_feature and current_type:
                if current_type not in feature_blocks:
                    feature_blocks[current_type] = []
                feature_blocks[current_type].append('\n'.join(current_feature))
            
            # Track feature variety
            structure['feature_variety'] = {k: len(v) for k, v in feature_blocks.items()}
            
            # Collect diverse samples (up to 5 of each type)
            for feature_type, blocks in feature_blocks.items():
                for block in blocks[:5]:  # Get up to 5 samples of each type
                    structure['feature_samples'].append(block)
                    if len(structure['feature_samples']) >= 20:
                        break
                if len(structure['feature_samples']) >= 20:
                    break
        
        # Extract more sequence sample (first 20 lines after ORIGIN)
        if origin_start > 0 and origin_start + 20 < len(lines):
            structure['sequence_sample'] = '\n'.join(lines[origin_start:origin_start + 20])
        
        return structure
    
    def generate_claude_parser(self, genbank_sample: str, selected_features: List[str]) -> Optional[str]:
        """Generate parser using Claude API with comprehensive structure analysis"""
        if not self.client:
            return None
        
        # Analyze file structure with more samples
        structure = self.analyze_genbank_structure(genbank_sample)
        
        # Create feature string for prompt
        feature_str = ', '.join(selected_features)
        
        # Show all available feature samples (up to 10)
        feature_samples_str = '\n\n'.join(structure['feature_samples'][:10])
        
        prompt = f"""Generate a Python function to parse GenBank files based on this comprehensive structure analysis:

FILE STRUCTURE:
- Total size: {structure['file_size']} bytes
- Total lines: {structure['total_lines']}
- Feature types found: {structure['feature_variety']}

HEADER SAMPLE (first 50 lines):
{structure['header_sample']}

FEATURE SAMPLES (showing {len(structure['feature_samples'][:10])} diverse examples):
{feature_samples_str}

SEQUENCE SECTION SAMPLE:
{structure['sequence_sample']}

REQUIREMENTS:
1. Function signature: def parse_genbank_custom(content: str, selected_features: List[str]) -> List[Dict]
2. Extract ONLY these feature types: {feature_str}
3. For each feature, extract ALL of these fields:
   - locus_tag (from /locus_tag="...")
   - gene name (from /gene="...")
   - product (from /product="...") - IMPORTANT: Handle multi-line products correctly!
   - start, end positions (handle join() and complement() correctly)
   - strand (+ or - based on complement)
   - protein_id (from /protein_id="...")
   - EC_number (from /EC_number="...")
   - NT sequence (extract from ORIGIN section using coordinates)
4. Use compiled regex patterns for maximum speed
5. Handle multi-line qualifier values correctly (product often spans multiple lines)
6. Handle both simple locations (123..456) and complex ones (join(123..456,789..1012))
7. Return format MUST be exactly:
   {{
       'Locus Tag': str,
       'Gene': str (use 'N/A' if missing),
       'EC Number': str (use 'N/A' if missing),
       'NT Seq': str (the actual DNA sequence),
       'Product': str,
       'Start': int,
       'End': int,
       'Strand': str ('+' or '-'),
       'Feature Type': str
   }}
8. Include all necessary imports (re, typing, etc) at the function level

Generate the fastest and most accurate parser that handles all edge cases shown in the samples."""

        try:
            # Use correct model name
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Fixed model name
                max_tokens=4000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            
            # Extract Python code
            code_match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
            if code_match:
                return code_match.group(1)
            
            # Try to find function definition directly
            if 'def parse_genbank_custom' in content:
                return content
                
        except Exception as e:
            logging.error(f"Claude API error: {e}")
            
        return None
    
    def get_optimized_regex_parser(self) -> str:
        """Get optimized regex-based parser for default use - with sequence extraction fixed"""
        return '''
def parse_genbank_custom(content: str, selected_features: List[str]) -> List[Dict]:
    """Optimized regex-based GenBank parser with proper sequence extraction"""
    import re
    from typing import List, Dict
    
    # Pre-compile patterns
    feature_pattern = re.compile(r'     (' + '|'.join(selected_features) + r')\\s+(?:complement\\()?(?:join\\()?([\\d,\\.\\s]+)')
    locus_pattern = re.compile(r'/locus_tag="([^"]+)"')
    gene_pattern = re.compile(r'/gene="([^"]+)"')
    product_pattern = re.compile(r'/product="([^"\\n]+(?:\\n\\s+[^/"]+)*)"', re.MULTILINE | re.DOTALL)
    protein_pattern = re.compile(r'/protein_id="([^"]+)"')
    ec_pattern = re.compile(r'/EC_number="([^"]+)"')
    
    # Find sections
    features_start = content.find('FEATURES')
    if features_start == -1:
        return []
    
    origin_start = content.find('ORIGIN', features_start)
    if origin_start == -1:
        origin_start = len(content)
    
    # Extract sequence data if present
    sequence_data = ""
    if origin_start < len(content):
        origin_section = content[origin_start:]
        # Find the end marker
        end_marker = origin_section.find('//')
        if end_marker > 0:
            origin_section = origin_section[:end_marker]
        
        # Remove ORIGIN line, numbers, spaces, and newlines
        sequence_lines = origin_section.split('\\n')[1:]  # Skip ORIGIN line
        for line in sequence_lines:
            # Remove line numbers and spaces
            cleaned_line = re.sub(r'^\\s*\\d+\\s*', '', line)
            cleaned_line = cleaned_line.replace(' ', '').strip()
            sequence_data += cleaned_line
        
        # Convert to uppercase
        sequence_data = sequence_data.upper()
    
    # Process features
    features_section = content[features_start:origin_start]
    
    # Split into blocks
    genes = []
    
    for match in feature_pattern.finditer(features_section):
        feature_type = match.group(1)
        location_str = match.group(2)
        
        # Parse location
        coords = re.findall(r'\\d+', location_str)
        if len(coords) >= 2:
            start = int(coords[0])
            end = int(coords[-1])
        else:
            continue
            
        # Check for complement
        feature_start_pos = match.start()
        line_start = features_section.rfind('\\n', 0, feature_start_pos)
        if line_start == -1:
            line_start = 0
        feature_line = features_section[line_start:feature_start_pos + 200]
        strand = '-' if 'complement' in feature_line else '+'
        
        # Find the end of this feature block
        next_feature = re.search(r'^\\s{5}\\w+\\s+', features_section[match.end():], re.MULTILINE)
        if next_feature:
            block = features_section[match.start():match.end() + next_feature.start()]
        else:
            block = features_section[match.start():]
        
        # Extract qualifiers
        locus_match = locus_pattern.search(block)
        locus_tag = locus_match.group(1) if locus_match else ''
        
        gene_match = gene_pattern.search(block)
        gene_name = gene_match.group(1) if gene_match else ''
        
        product_match = product_pattern.search(block)
        if product_match:
            # Clean up multi-line product
            product = product_match.group(1)
            product = re.sub(r'\\n\\s+', ' ', product).strip()
        else:
            product = ''
        
        protein_match = protein_pattern.search(block)
        protein_id = protein_match.group(1) if protein_match else ''
        
        ec_match = ec_pattern.search(block)
        ec_number = ec_match.group(1) if ec_match else ''
        
        # Extract sequence
        nt_seq = ''
        if sequence_data and start > 0 and end <= len(sequence_data):
            nt_seq = sequence_data[start-1:end]
            if strand == '-':
                # Reverse complement
                complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
                nt_seq = ''.join(complement.get(base, base) for base in nt_seq[::-1])
        
        # Create gene entry in required format
        gene_entry = {
            'Locus Tag': locus_tag,
            'Gene': gene_name if gene_name else 'N/A',
            'EC Number': ec_number if ec_number else 'N/A',
            'NT Seq': nt_seq,
            'Product': product,
            'Start': start,
            'End': end,
            'Strand': strand,
            'Feature Type': feature_type
        }
        
        # Only add if has locus tag or gene name
        if locus_tag or gene_name:
            genes.append(gene_entry)
    
    return genes
'''
    
    def parse_with_biopython(self, genbank_data: str, selected_features: List[str], 
                           progress_callback=None) -> List[Dict]:
        """Parse using BioPython (default method)"""
        from Bio import SeqIO
        from io import StringIO
        
        if progress_callback:
            progress_callback(20, "Parsing with BioPython...")
        
        try:
            records = list(SeqIO.parse(StringIO(genbank_data), "genbank"))
            if not records:
                return []
            
            record = records[0]
            genes = []
            
            if progress_callback:
                progress_callback(50, f"Processing {len(record.features)} features...")
            
            for i, feature in enumerate(record.features):
                if feature.type not in selected_features:
                    continue
                
                # Progress update every 100 features
                if i % 100 == 0 and progress_callback:
                    progress = 50 + int((i / len(record.features)) * 40)
                    progress_callback(progress, f"Processing feature {i}/{len(record.features)}...")
                
                # Extract qualifiers
                qualifiers = feature.qualifiers
                
                # Get location info
                start = int(feature.location.start) + 1  # Convert to 1-based
                end = int(feature.location.end)
                strand = '+' if feature.location.strand == 1 else '-'
                
                # Extract sequence - IMPORTANT: This is what ensures sequences are included
                try:
                    seq_obj = feature.extract(record.seq)
                    nt_seq = str(seq_obj)
                except:
                    nt_seq = ''
                
                # Create entry in required format
                gene_entry = {
                    'Locus Tag': qualifiers.get('locus_tag', [''])[0],
                    'Gene': qualifiers.get('gene', ['N/A'])[0],
                    'EC Number': qualifiers.get('EC_number', ['N/A'])[0],
                    'NT Seq': nt_seq,  # This field must be populated!
                    'Product': qualifiers.get('product', [''])[0],
                    'Start': start,
                    'End': end,
                    'Strand': strand,
                    'Feature Type': feature.type
                }
                
                # Only add if has locus tag or gene name
                if gene_entry['Locus Tag'] or gene_entry['Gene'] != 'N/A':
                    genes.append(gene_entry)
            
            if progress_callback:
                progress_callback(90, f"Completed! Found {len(genes)} genes")
            
            # Log sequence extraction status
            genes_with_seq = sum(1 for g in genes if g.get('NT Seq', ''))
            logging.info(f"Extracted {len(genes)} genes, {genes_with_seq} with sequences")
            
            return genes
            
        except Exception as e:
            logging.error(f"BioPython parsing error: {e}")
            return []
    
    def parse_with_claude(self, genbank_data: str, selected_features: List[str], 
                         api_key: str = None, progress_callback=None) -> List[Dict]:
        """Parse using Claude-generated parser"""
        
        if progress_callback:
            progress_callback(10, "Initializing Claude parser...")
        
        # Use provided API key or default from api.py
        if not self.client:
            key_to_use = api_key or ANTHROPIC_API_KEY
            if key_to_use:
                self.init_claude_client(key_to_use)
        
        if not self.client:
            logging.warning("Claude client not available, falling back to regex parser")
            parser_code = self.get_optimized_regex_parser()
        else:
            if progress_callback:
                progress_callback(20, "Generating custom parser with Claude...")
            
            # Send more comprehensive sample to Claude
            sample_size = min(20000, len(genbank_data))  # Send up to 20KB
            parser_code = self.generate_claude_parser(genbank_data[:sample_size], selected_features)
            
            if not parser_code:
                logging.warning("Failed to generate Claude parser, using optimized regex")
                parser_code = self.get_optimized_regex_parser()
            else:
                if progress_callback:
                    progress_callback(40, "Claude parser generated successfully!")
                
                # Log the generated parser for debugging
                logging.debug(f"Generated parser code:\n{parser_code[:500]}...")
        
        # Execute the parser
        if progress_callback:
            progress_callback(60, "Executing parser...")
        
        try:
            # Create namespace with necessary imports
            namespace = {
                'List': List,
                'Dict': Dict,
                're': re,
                'typing': {'List': List, 'Dict': Dict}
            }
            exec(parser_code, namespace)
            
            if 'parse_genbank_custom' in namespace:
                start_time = time.time()
                genes = namespace['parse_genbank_custom'](genbank_data, selected_features)
                parse_time = time.time() - start_time
                
                if progress_callback:
                    progress_callback(90, f"Parsed {len(genes)} genes in {parse_time:.2f}s")
                
                # Log sequence extraction status
                genes_with_seq = sum(1 for g in genes if g.get('NT Seq', ''))
                logging.info(f"Parser extracted {len(genes)} genes, {genes_with_seq} with sequences")
                
                return genes
            else:
                raise ValueError("Parser function not found in generated code")
                
        except Exception as e:
            logging.error(f"Parser execution error: {e}")
            # Fallback to BioPython
            logging.info("Falling back to BioPython parser")
            return self.parse_with_biopython(genbank_data, selected_features, progress_callback)