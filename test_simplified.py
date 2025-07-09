#!/usr/bin/env python3
"""
Test script for simplified GenBank Gene Extractor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all core modules can be imported"""
    try:
        from core.genbank_processor import GenBankProcessor
        from core.kegg_integration import KEGGIntegration
        from core.staged_processor import StagedGenomeProcessor
        from core.smart_parser_generator import SmartParserGenerator
        print("✅ All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external API calls"""
    try:
        from core.genbank_processor import GenBankProcessor
        processor = GenBankProcessor()
        
        # Test basic methods exist
        assert hasattr(processor, 'search_assembly')
        assert hasattr(processor, 'fetch_genbank')
        assert hasattr(processor, 'parse_genes')
        
        print("✅ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_kegg_integration():
    """Test KEGG integration setup"""
    try:
        from core.kegg_integration import KEGGIntegration
        kegg = KEGGIntegration()
        
        # Test basic methods exist
        assert hasattr(kegg, 'search_organism_genomes')
        assert hasattr(kegg, 'get_genome_details')
        
        print("✅ KEGG integration test passed")
        return True
    except Exception as e:
        print(f"❌ KEGG integration test failed: {e}")
        return False

def test_staged_processor():
    """Test staged processor setup"""
    try:
        from core.staged_processor import StagedGenomeProcessor
        processor = StagedGenomeProcessor()
        
        # Test basic methods exist
        assert hasattr(processor, 'stage1_download_all_genbank_files')
        assert hasattr(processor, 'stage2_generate_summary_report')
        assert hasattr(processor, 'stage3_batch_parse_files')
        
        print("✅ Staged processor test passed")
        return True
    except Exception as e:
        print(f"❌ Staged processor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Simplified GenBank Gene Extractor")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_kegg_integration,
        test_staged_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The simplified interface should work correctly.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 