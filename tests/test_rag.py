import sys
import os
import pytest

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import RAGComponent

@pytest.fixture
def temp_recipes_dir(tmp_path):
    #Create a temporary directory with dummy recipe files.
    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    
    # Create a dummy SQL injection recipe
    sql_recipe = recipes_dir / "CWE-89-SQL_Injection.txt"
    sql_recipe.write_text("Avoid using string concatenation for SQL queries. Use parameterized queries instead.")
    
    # Create a dummy XSS recipe
    xss_recipe = recipes_dir / "CWE-79-XSS.txt"
    xss_recipe.write_text("Sanitize user input before rendering it in the browser to prevent Cross-Site Scripting.")
    
    return str(recipes_dir)

def test_rag_retrieval_logic(temp_recipes_dir):
    #Test RAG retrieval logic - validates that relevant recipes are retrieved correctly.
    rag = RAGComponent(recipes_dir=temp_recipes_dir)
    
    # Test that index is built
    assert rag.index is not None
    assert rag.index.ntotal == 2
    assert len(rag.documents) == 2
    
    # Test retrieval for SQL Injection query
    query_sql = "CWE-89 python SELECT * FROM users WHERE name = 'admin'"
    results_sql = rag.retrieve(query_sql, k=1)
    assert len(results_sql) == 1
    assert "SQL" in results_sql[0] or "parameterized" in results_sql[0]
    
    # Test retrieval for XSS query
    query_xss = "CWE-79 javascript <script>alert(1)</script>"
    results_xss = rag.retrieve(query_xss, k=1)
    assert len(results_xss) == 1
    assert "XSS" in results_xss[0] or "Sanitize" in results_xss[0]
