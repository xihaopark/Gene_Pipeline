# core/genbank_processor.py
import requests
import pandas as pd
import re
import logging
from io import StringIO
from Bio import SeqIO
from typing import List, Dict, Optional
from .web_file_manager import WebFileManager
import warnings
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)

class GenBankProcessor:
    """Web-based GenBank processor for gene extraction"""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.file_manager = WebFileManager()
        # Create a session with SSL verification disabled
        self.session = requests.Session()
        self.session.verify = False
    
    def fetch_ec_from_protein(self, protein_id: str) -> str:
        """
        Fetch EC number from protein ID if not available locally
        """
        if not protein_id or protein_id == "N/A":
            return ""
        
        url = f"{self.base_url}efetch.fcgi"
        params = {
            "db": "protein",
            "id": protein_id,
            "retmode": "xml"
        }
        
        try:
            resp = self.session.get(url, params=params)
            resp.raise_for_status()
            xml_text = resp.text
            pattern = (
                r"<GBQualifier_name>EC_number</GBQualifier_name>\s*"
                r"<GBQualifier_value>([^<]+)</GBQualifier_value>"
            )
            match = re.search(pattern, xml_text)
            if match:
                return match.group(1).strip()
            else:
                return ""
        except Exception as ex:
            logging.error(f"Error fetching EC number for protein {protein_id}: {ex}")
            return ""
    
    def search_assembly(self, accession: str) -> List[Dict]:
        """Search for assembly by accession"""
        # Check if it's a direct accession
        if re.match(r'^(GCF|GCA)_\d+\.\d+$', accession):
            term = f"{accession}[Assembly Accession]"
        else:
            term = accession
        
        # ESearch
        url = f"{self.base_url}esearch.fcgi"
        params = {
            "db": "assembly",
            "term": term,
            "retmode": "json",
            "retmax": 20
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                return []
            
            return self._fetch_assembly_details(id_list)
            
        except Exception as e:
            logging.error(f"Error searching assembly: {e}")
            return []
    
    def _fetch_assembly_details(self, id_list: List[str]) -> List[Dict]:
        """Fetch assembly details using ESummary"""
        url = f"{self.base_url}esummary.fcgi"
        params = {
            "db": "assembly",
            "id": ",".join(id_list),
            "retmode": "json"
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            assemblies = []
            result = data.get("result", {})
            uids = result.get("uids", [])
            
            for uid in uids:
                if uid == "uids":
                    continue
                record = result.get(uid, {})
                assemblies.append({
                    "assembly_id": uid,
                    "assembly_acc": record.get("assemblyaccession", ""),
                    "organism": record.get("organism", ""),
                    "assembly_name": record.get("assemblyname", ""),
                    "submitter": record.get("submitter", ""),
                    "assembly_level": record.get("assemblylevel", ""),
                    "version": record.get("assemblyversion", "")
                })
            
            return assemblies
            
        except Exception as e:
            logging.error(f"Error fetching assembly details: {e}")
            return []
    
    def link_to_nuccore(self, assembly_accession: str) -> List[Dict]:
        """Link assembly to nucleotide sequences"""
        # First get assembly ID
        search_url = f"{self.base_url}esearch.fcgi"
        search_params = {
            "db": "assembly",
            "term": f"{assembly_accession}[Assembly Accession]",
            "retmode": "json"
        }
        
        try:
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            data = response.json()
            idlist = data.get("esearchresult", {}).get("idlist", [])
            
            if not idlist:
                return []
            
            assembly_id = idlist[0]
            
            # Link to nuccore
            link_url = f"{self.base_url}elink.fcgi"
            link_params = {
                "dbfrom": "assembly",
                "db": "nuccore",
                "id": assembly_id,
                "retmode": "json"
            }
            
            response = self.session.get(link_url, params=link_params)
            response.raise_for_status()
            link_data = response.json()
            
            nuccore_ids = []
            linksets = link_data.get("linksets", [])
            for ls in linksets:
                linksetdbs = ls.get("linksetdbs", [])
                for linksetdb in linksetdbs:
                    if linksetdb.get("dbto") == "nuccore":
                        nuccore_ids.extend(linksetdb.get("links", []))
            
            if not nuccore_ids:
                return []
            
            # Get nuccore details
            return self._fetch_nuccore_details(nuccore_ids)
            
        except Exception as e:
            logging.error(f"Error linking to nuccore: {e}")
            return []
    
    def _fetch_nuccore_details(self, nuccore_ids: List[str]) -> List[Dict]:
        """Fetch nucleotide sequence details"""
        url = f"{self.base_url}esummary.fcgi"
        params = {
            "db": "nuccore",
            "id": ",".join(nuccore_ids[:50]),  # Limit to first 50
            "retmode": "json"
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            records = []
            result = data.get("result", {})
            uids = result.get("uids", [])
            
            for uid in uids:
                if uid == "uids":
                    continue
                record = result.get(uid, {})
                records.append({
                    "nuccore_id": uid,
                    "accver": record.get("accessionversion", ""),
                    "title": record.get("title", ""),
                    "length": int(record.get("slen", 0))
                })
            
            return records
            
        except Exception as e:
            logging.error(f"Error fetching nuccore details: {e}")
            return []
    
    def fetch_genbank(self, accession: str, save_to_cache: bool = True) -> Optional[str]:
        """Fetch GenBank record with full annotations and optionally save to cache"""
        # Check if file exists in cache first
        if save_to_cache and self.file_manager.genbank_exists(accession):
            logging.info(f"Loading existing GenBank from cache: {accession}")
            return self.file_manager.get_genbank_content(accession)
        
        # Try multiple approaches to get complete GenBank record
        genbank_data = None
        
        # Method 1: Try with gbwithparts to get complete record
        logging.info(f"Attempting to download complete GenBank record for {accession}")
        url = f"{self.base_url}efetch.fcgi"
        
        # First try: Get full GenBank with features
        params = {
            "db": "nuccore",
            "id": accession,
            "rettype": "gbwithparts",  # Include all parts and features
            "retmode": "text"
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            content = response.text
            
            # Check if we got a substantial record
            if content and "FEATURES" in content:
                logging.info(f"Successfully downloaded complete GenBank record with features")
                genbank_data = content
            else:
                logging.warning("First attempt did not return features, trying alternative method")
        except Exception as e:
            logging.error(f"First download attempt failed: {e}")
        
        # Method 2: If first method failed, try standard gb format
        if not genbank_data:
            params = {
                "db": "nuccore",
                "id": accession,
                "rettype": "gb",
                "retmode": "text"
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                content = response.text
                
                if content:
                    logging.info("Downloaded GenBank record with standard method")
                    genbank_data = content
            except Exception as e:
                logging.error(f"Second download attempt failed: {e}")
        
        # Method 3: If still no features, try to get the RefSeq version
        if genbank_data and "FEATURES" not in genbank_data:
            logging.warning("Downloaded record has no FEATURES section, trying RefSeq version")
            
            # Convert NZ_ to regular accession if needed
            alt_accession = accession.replace("NZ_", "") if accession.startswith("NZ_") else f"NZ_{accession}"
            
            params = {
                "db": "nuccore",
                "id": alt_accession,
                "rettype": "gbwithparts",
                "retmode": "text"
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                alt_content = response.text
                
                if alt_content and "FEATURES" in alt_content:
                    logging.info(f"Found features in alternative accession: {alt_accession}")
                    genbank_data = alt_content
                    accession = alt_accession  # Update accession for caching
            except Exception as e:
                logging.error(f"Alternative accession download failed: {e}")
        
        # Check final result
        if genbank_data:
            feature_count = genbank_data.count("     gene            ")
            logging.info(f"Final GenBank record contains approximately {feature_count} gene features")
            
            # Save to cache if requested and content is good
            if save_to_cache and "FEATURES" in genbank_data:
                self.file_manager.save_genbank_content(accession, genbank_data)
            
            return genbank_data
        else:
            logging.error("Failed to download GenBank record with all methods")
            return None
    
    def analyze_genbank_structure(self, genbank_data: str):
        """
        Analyze GenBank data structure and return a record object.
        Handles both XML and text formats with detailed diagnostics.
        """
        data = genbank_data.strip()
        
        # Log first few lines for debugging
        lines = data.split('\n')[:10]
        logging.info(f"GenBank data preview: {lines[:5]}")
        
        # Check for FEATURES section
        has_features = "FEATURES" in data
        has_origin = "ORIGIN" in data
        feature_count = data.count("     gene            ")
        cds_count = data.count("     CDS             ")
        
        logging.info(f"GenBank analysis: Features={has_features}, Origin={has_origin}, Genes≈{feature_count}, CDS≈{cds_count}")
        
        if not has_features:
            logging.warning("⚠️  GenBank record has no FEATURES section - this explains why no genes were found!")
            logging.warning("This is likely a 'CON' (contig) record with only sequence data")
            
        if data.startswith("<?xml"):
            # XML format - use GBSeq parser
            try:
                from Bio.NCBI.GBSeq import parse as gbseq_parse
                records = list(gbseq_parse(StringIO(genbank_data)))
                if not records:
                    return None
                record = records[0]
                logging.info(f"Parsed XML format record: {type(record)}")
                return record
            except ImportError:
                logging.error("Bio.NCBI.GBSeq module is not available. Please update your Biopython version.")
                return None
        elif data.startswith("LOCUS"):
            # Text format - use SeqIO
            try:
                records = list(SeqIO.parse(StringIO(genbank_data), "genbank"))
                if not records:
                    logging.error("No records found in GenBank text")
                    return None
                record = records[0]
                logging.info(f"Parsed text format record: {type(record)}")
                logging.info(f"Record contains {len(record.features)} features")
                
                # Analyze feature types
                feature_types = {}
                for feature in record.features:
                    ftype = feature.type
                    feature_types[ftype] = feature_types.get(ftype, 0) + 1
                
                logging.info(f"Feature breakdown: {feature_types}")
                return record
                
            except Exception as e:
                logging.error(f"Error parsing GenBank text format: {e}")
                return None
        else:
            logging.error("Input data does not appear to be valid GenBank format.")
            logging.error(f"Data starts with: {data[:100]}")
            return None

    def parse_genes(self, genbank_data: str) -> List[Dict]:
        """Parse genes from GenBank data using BioPython with proper format detection"""
        record = self.analyze_genbank_structure(genbank_data)
        if record is None:
            logging.error("Failed to parse GenBank data")
            return []
        
        genes = []
        gene_dict = {}  # For deduplication by locus_tag
        
        logging.info(f"Record ID: {getattr(record, 'id', 'N/A')}")
        logging.info(f"Number of features: {len(getattr(record, 'features', []))}")
        
        # Handle both Bio.SeqRecord and GBSeq formats
        features = getattr(record, 'features', [])
        if not features and hasattr(record, 'feature_table'):
            # Handle GBSeq format
            features = getattr(record, 'feature_table', [])
        
        for feature in features:
            # Handle different feature formats
            feature_type = getattr(feature, 'type', None)
            if hasattr(feature, 'key'):
                feature_type = getattr(feature, 'key')
            
            if feature_type not in ("gene", "CDS"):
                continue
            
            # Handle qualifiers in different formats
            qualifiers = {}
            if hasattr(feature, 'qualifiers'):
                qualifiers = feature.qualifiers
            elif hasattr(feature, 'quals'):
                # Convert GBSeq quals format to standard format
                quals_list = getattr(feature, 'quals', [])
                for qual in quals_list:
                    name = getattr(qual, 'name', '')
                    value = getattr(qual, 'value', '')
                    if name:
                        if name in qualifiers:
                            if isinstance(qualifiers[name], list):
                                qualifiers[name].append(value)
                            else:
                                qualifiers[name] = [qualifiers[name], value]
                        else:
                            qualifiers[name] = [value]
            
            # Ensure all qualifier values are lists
            for key, value in qualifiers.items():
                if not isinstance(value, list):
                    qualifiers[key] = [value]
            
            # Get locus tag as key
            locus_tag = qualifiers.get("locus_tag", [None])[0]
            if not locus_tag:
                locus_tag = qualifiers.get("gene", [f"unknown_{len(genes)}"])[0]
            
            # Handle location information
            start, end, strand = 0, 0, "."
            if hasattr(feature, 'location'):
                start = int(feature.location.start) + 1  # 1-based
                end = int(feature.location.end)
                strand = "+" if feature.location.strand == 1 else "-" if feature.location.strand == -1 else "."
            elif hasattr(feature, 'intervals'):
                # Handle GBSeq intervals format
                intervals = getattr(feature, 'intervals', [])
                if intervals:
                    interval = intervals[0]  # Use first interval
                    start = int(getattr(interval, 'from', 0))
                    end = int(getattr(interval, 'to', 0))
                    # For GBSeq, coordinates are already 1-based
            
            # Extract gene information FIRST
            gene_info = {
                "locus_tag": locus_tag,
                "gene": qualifiers.get("gene", [""])[0],
                "product": qualifiers.get("product", [""])[0],
                "start": start,
                "end": end,
                "strand": strand,
                "protein_id": qualifiers.get("protein_id", [""])[0],
                "ec_number": "",  # Will be filled below
                "length": end - start + 1 if end > start else 0,
                "feature_type": feature_type
            }
            
            # Get EC number with fallback to API if needed - AFTER gene_info is created
            ec_number = ""
            if "EC_number" in qualifiers and qualifiers["EC_number"][0]:
                ec_number = qualifiers["EC_number"][0]
            elif gene_info.get("protein_id"):
                # Try to fetch from protein database
                ec_number = self.fetch_ec_from_protein(gene_info["protein_id"])
            
            # Update the EC number in gene_info
            gene_info["ec_number"] = ec_number
            
            # Extract sequence if possible
            if hasattr(feature, 'extract') and hasattr(record, 'seq'):
                try:
                    seq_obj = feature.extract(record.seq)
                    gene_info["sequence"] = str(seq_obj)
                except:
                    gene_info["sequence"] = ""
            else:
                gene_info["sequence"] = ""
            
            # Add translation for CDS
            if feature_type == "CDS":
                gene_info["translation"] = qualifiers.get("translation", [""])[0]
            else:
                gene_info["translation"] = ""
            
            # Deduplication: prefer CDS over gene
            if locus_tag in gene_dict:
                if feature_type == "CDS":
                    gene_dict[locus_tag] = gene_info
            else:
                gene_dict[locus_tag] = gene_info
        
        result = list(gene_dict.values())
        logging.info(f"Extracted {len(result)} unique genes")
        return result
    
    def genes_to_csv(self, genes: List[Dict], accession: str = None) -> str:
        """Convert gene list to CSV string"""
        if not genes:
            return ""
        
        df = pd.DataFrame(genes)
        
        # Reorder columns for better readability
        column_order = [
            "locus_tag", "gene", "product", "start", "end", "strand", 
            "length", "protein_id", "ec_number", "feature_type", 
            "sequence", "translation"
        ]
        
        # Only include columns that exist
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        return df.to_csv(index=False)
    
    def save_genes_to_cache(self, genes: List[Dict], accession: str) -> bool:
        """Save genes to cache as CSV"""
        if not genes:
            return False
        
        csv_content = self.genes_to_csv(genes, accession)
        return self.file_manager.save_csv_content(accession, csv_content)