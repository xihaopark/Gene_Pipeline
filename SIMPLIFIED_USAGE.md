# Simplified GenBank Gene Extractor - Usage Guide

## 🎯 Overview

The interface has been simplified to provide a streamlined, single-page experience with AI-first processing. No more complex tabs or configuration options - just search, process, and analyze!

## 🚀 New Simplified Workflow

### 1. **Single Search Interface**
- **Two-column layout** with side-by-side search options
- **Left column**: Search by organism name (KEGG database)
- **Right column**: Search by accession number (NCBI database)

### 2. **AI-First Processing**
- **No parser selection needed** - Claude AI is used by default
- **Faster and more accurate** than traditional BioPython parsing
- **Intelligent structure analysis** for optimal gene extraction

### 3. **Integrated Sequence Mapping**
- **No separate tab** - mapping appears automatically after processing
- **Context-aware options** based on your results
- **Seamless workflow** from search to analysis

## 📋 Step-by-Step Usage

### Option A: Search by Organism Name

1. **Enter organism name** in the left column
   - Example: `Clostridioides difficile`
   - Uses KEGG database for comprehensive genome search

2. **Set parameters**
   - Max Genomes: 1-20 (default: 5)
   - System automatically optimizes for best results

3. **Click "🔍 Search by Organism"**
   - Finds and downloads multiple genomes
   - Shows progress and results summary

4. **Review results and process**
   - See genome count, total genes, file sizes
   - Choose to include DNA sequences
   - Click "🚀 Process All Genomes"

### Option B: Search by Accession Number

1. **Enter accession number** in the right column
   - Example: `GCA_000005825.2`
   - Direct NCBI database search

2. **Click "🔍 Search by Accession"**
   - Finds available assemblies
   - Shows assembly details

3. **Select assembly and process**
   - Choose from available assemblies
   - Click "🚀 Process Selected Assembly"

## 📊 Results & Analysis

### Processing Results
- **Automatic AI parsing** with Claude
- **Gene count and success metrics**
- **Individual file results** (for batch processing)
- **Downloadable CSV files**

### Integrated Sequence Mapping
- **Appears automatically** after processing
- **For KEGG results**: Multi-genome comparison
- **For direct results**: Upload additional files for comparison
- **Interactive visualizations** and downloadable results

## 🎨 Interface Improvements

### Visual Enhancements
- **Emoji icons** for better visual hierarchy
- **Clear progress indicators** with success/error states
- **Responsive layout** that works on different screen sizes
- **Contextual help text** for all input fields

### Simplified Controls
- **No complex configuration panels**
- **Smart defaults** for all parameters
- **One-click processing** with minimal user input
- **Clear action buttons** with descriptive labels

## 🔧 Advanced Features (Still Available)

### AI Processing
- **Automatic structure analysis** of GenBank files
- **Optimized parsing** for different genome types
- **Error handling** and retry mechanisms
- **Performance optimization** for large datasets

### Sequence Analysis
- **Clustering-based matching** for faster comparisons
- **Configurable similarity thresholds**
- **Multiple visualization options**
- **Export capabilities** for further analysis

## 💡 Tips for Best Results

### For Organism Search
- Use **scientific names** for best results
- Start with **smaller genome counts** (2-5) for testing
- Enable **"Include DNA sequences"** for sequence mapping

### For Accession Search
- Use **complete accession numbers** (e.g., GCA_000005825.2)
- Check **assembly details** before processing
- Consider **assembly level** (Complete > Chromosome > Scaffold)

### For Sequence Mapping
- Ensure **sequences are included** in your processed data
- Use **appropriate similarity thresholds** (0.8 is usually good)
- Enable **clustering** for faster processing of large datasets

## 🚨 Common Issues & Solutions

### Search Issues
- **No results found**: Check spelling of organism name
- **Too many results**: Reduce max genome count
- **Network errors**: Check internet connection

### Processing Issues
- **AI parsing fails**: Ensure API key is configured
- **Memory errors**: Reduce genome count or disable sequences
- **Slow processing**: Enable clustering for large datasets

### Mapping Issues
- **No sequences available**: Re-run with "Include DNA sequences"
- **No matches found**: Lower similarity threshold
- **Slow mapping**: Enable clustering option

## 🔄 Migration from Old Interface

### What Changed
- **Tabs removed**: Everything on one page
- **Parser selection removed**: AI is default
- **Sequence mapping integrated**: No separate tab needed
- **Simplified controls**: Fewer options, smarter defaults

### What Stayed the Same
- **All core functionality** is preserved
- **Same output formats** (CSV files)
- **Same analysis capabilities**
- **Same API integrations** (KEGG, NCBI, Claude)

## 🎯 Benefits of Simplified Interface

1. **Faster workflow** - Less clicking, more results
2. **Better user experience** - Clear, intuitive interface
3. **Reduced complexity** - No overwhelming options
4. **AI-powered** - Better results with less configuration
5. **Integrated analysis** - Seamless from search to mapping

The simplified interface maintains all the power of the original system while making it much more accessible and user-friendly! 