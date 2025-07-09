# GenBank Gene Extractor 2.0

A powerful genomic data processing tool for extracting gene information from GenBank database with advanced sequence analysis capabilities.

## 🌟 Features

- **KEGG Batch Search**: Search and download multiple genomes from KEGG database
- **Direct Accession Search**: Search by specific genome accession numbers (GCA/GCF)
- **Sequence Mapping**: Advanced sequence comparison and clustering analysis
- **AI-Enhanced Parsing**: Claude AI-powered intelligent GenBank parser generation
- **Web Interface**: User-friendly Streamlit-based interface
- **Batch Processing**: Three-stage workflow (Download → Preview → Parse)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Stable internet connection (for accessing NCBI and KEGG databases)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/xihaopark/Gene_Pipeline.git
cd Gene_Pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys (optional, for AI parsing):
```bash
cp api.example.py api.py
# Edit api.py with your actual API keys
```

4. Run the application:
```bash
python run.py
```

Or use Streamlit directly:
```bash
streamlit run main.py
```

## 📖 Usage

### 1. KEGG Batch Search

1. Enter organism name (e.g., "Clostridioides difficile")
2. Set parameters (max genomes, include sequences)
3. Stage 1: Download GenBank files
4. Stage 2: Preview results
5. Stage 3: Parse and generate CSV tables

### 2. Direct Accession Search

1. Enter genome accession number (e.g., GCA_000005825.2)
2. Search and select assembly
3. Download and extract genes
4. Export results

### 3. Sequence Mapping

1. Upload multiple CSV files with gene sequences
2. Configure similarity threshold and clustering options
3. Run mapping analysis
4. View results with interactive visualizations

## 🏗️ System Architecture

```
Gene_Pipeline/
├── main.py                 # Main Streamlit application
├── run.py                  # Server startup script
├── api.example.py          # API key template
├── requirements.txt        # Dependencies
└── core/                   # Core modules
    ├── genbank_processor.py      # GenBank data processing
    ├── kegg_integration.py       # KEGG database integration
    ├── staged_processor.py       # Three-stage processing
    ├── sequence_analyzer.py      # Sequence analysis & clustering
    ├── smart_parser_generator.py # AI-enhanced parser
    └── ...
```

## 🔧 Configuration

### API Keys

The application supports multiple API key sources:

1. **api.py file** (recommended for local development)
2. **Environment variables**
3. **Streamlit secrets** (for cloud deployment)
4. **.env file**

### File Storage

- Default download folder: `~/Downloads`
- Project folders: `GenBank_Analysis_[project_name]/`
- Configurable through the web interface

## 📊 Output Format

Generated CSV files include:

- **Locus Tag**: Gene locus identifier
- **Gene**: Gene name
- **Product**: Gene product description
- **Start/End**: Gene positions
- **Strand**: DNA strand direction (+/-)
- **EC Number**: Enzyme commission number
- **NT Seq**: Nucleotide sequence (optional)
- **Protein ID**: Protein identifier

## 🚀 Deployment

### Local Development

```bash
python run.py
```

### Server Deployment

```bash
# Background execution
nohup python run.py --port 8501 > app.log 2>&1 &

# With screen
screen -S genbank_app
python run.py
```

### Docker Deployment

```bash
# Build image
docker build -t genbank-extractor .

# Run container
docker run -p 8501:8501 genbank-extractor
```

## 🔬 Advanced Features

### AI-Enhanced Parsing

- Uses Claude AI to analyze GenBank file structure
- Generates optimized parsing code
- Handles complex feature locations
- Supports multiple feature types

### Sequence Clustering

- K-means and DBSCAN clustering algorithms
- K-mer feature extraction
- Performance optimization for large datasets
- Interactive visualizations

### Batch Processing

- Parallel processing capabilities
- Progress tracking and error handling
- Automatic retry mechanisms
- Comprehensive logging

## 🛠️ Development

### Project Structure

- `core/`: Core functionality modules
- `oldfiles/`: Legacy code (deprecated)
- `yang/`: Sequence comparison utilities
- `GenBank_Analysis_/`: Sample data and outputs

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📋 Requirements

- Python 3.8+
- 2GB+ RAM (4GB+ recommended)
- 1GB+ disk space
- Internet connection for database access

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**: Use `--port` to specify different port
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **API rate limits**: Built-in delay mechanisms handle this automatically
4. **Large file processing**: Use clustering mode for better performance

### Debug Mode

Enable debug mode by setting environment variable:
```bash
export DEBUG_MODE=true
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Support

- GitHub Issues: [Report bugs or request features](https://github.com/xihaopark/Gene_Pipeline/issues)
- Documentation: See `README_CN.md` for Chinese documentation
- Email: Contact the maintainers

## 🙏 Acknowledgments

- NCBI for GenBank database access
- KEGG for genome database integration
- Anthropic for Claude AI API
- BioPython community for sequence analysis tools

---

**Note**: This tool is for research purposes. Please respect database usage policies and rate limits. 