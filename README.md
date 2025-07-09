# GenBank Gene Extractor 2.0

A powerful genomic data processing tool for extracting gene information from GenBank database with advanced sequence analysis capabilities.

## 🌟 Features

- **Unified Search Interface**: Single-page design with side-by-side organism and accession search
- **AI-First Processing**: Claude AI parsing enabled by default for optimal results
- **Integrated Workflow**: Seamless flow from search to analysis to sequence mapping
- **Smart Defaults**: Minimal configuration needed - just search and process
- **KEGG & NCBI Integration**: Comprehensive genome database access
- **Advanced Sequence Analysis**: Clustering-based mapping with interactive visualizations

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

### Simplified Single-Page Interface

**Left Column - Organism Search:**
1. Enter organism name (e.g., "Clostridioides difficile")
2. Set max genomes (1-20)
3. Click "🔍 Search by Organism"
4. Review results and click "🚀 Process All Genomes"

**Right Column - Accession Search:**
1. Enter genome accession (e.g., GCA_000005825.2)
2. Click "🔍 Search by Accession"
3. Select assembly and click "🚀 Process Selected Assembly"

**Integrated Sequence Mapping:**
- Appears automatically after processing
- Multi-genome comparison for KEGG results
- Upload additional files for single-genome comparison
- Interactive visualizations and downloadable results

For detailed usage instructions, see [SIMPLIFIED_USAGE.md](SIMPLIFIED_USAGE.md)

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