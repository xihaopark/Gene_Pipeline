# Installation Guide

## Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/xihaopark/Gene_Pipeline.git
cd Gene_Pipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys (Optional)

For AI-enhanced parsing features:

```bash
cp api.example.py api.py
```

Then edit `api.py` with your actual API keys:

```python
# Get your API key from https://console.anthropic.com/
ANTHROPIC_API_KEY = "your_actual_api_key_here"
```

### 4. Run the Application

```bash
python run.py
```

The application will be available at `http://localhost:8501`

## Detailed Configuration

### Environment Variables

You can also configure API keys using environment variables:

```bash
export ANTHROPIC_API_KEY="your_api_key_here"
export DEBUG_MODE="true"  # Optional: Enable debug mode
```

### Streamlit Secrets (for deployment)

Create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your_api_key_here"
```

### Docker Deployment

```bash
# Build image
docker build -t genbank-extractor .

# Run container
docker run -p 8501:8501 genbank-extractor
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Port already in use**: Specify a different port
   ```bash
   python run.py --port 8080
   ```

3. **Permission denied**: Check file permissions
   ```bash
   chmod +x run.py
   ```

### System Requirements

- Python 3.8 or higher
- 2GB+ RAM (4GB+ recommended for large datasets)
- 1GB+ available disk space
- Internet connection for database access

## First Run

1. Open your browser and navigate to `http://localhost:8501`
2. Try the "Direct Search" tab with a test accession: `GCA_000005825.2`
3. For KEGG search, try organism name: `Clostridioides difficile`
4. Check the sequence mapping tab with sample CSV files

## Getting Help

- Check the [README.md](README.md) for usage instructions
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for server deployment
- Open an issue on GitHub for bug reports or feature requests 