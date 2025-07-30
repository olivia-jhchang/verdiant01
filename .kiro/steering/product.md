# Product Overview

## Intelligent Search System for Public Documents

This is an intelligent search system designed for installation on internal servers of public institutions. The system connects to internal RDB databases to automatically collect, vectorize, and index administrative, civil affairs, and regulatory documents, providing RAG-based search services.

### Key Features

- **Internal-only operation**: No external network dependencies or web crawling - uses only internal database connections for security
- **Korean language optimization**: Specialized processing for Korean public documents with semantic chunking and Korean-specific NLP
- **RAG-based search**: Vector similarity search combined with LLM response generation for accurate, contextual answers
- **Document type classification**: Handles three main document types - administrative documents (행정문서), civil affairs documents (민원문서), and regulatory documents (조례문서)
- **Automated evaluation**: Built-in AutoEval system for continuous performance monitoring and quality assurance

### Target Users

- **System administrators**: Configure database connections, monitor system performance, manage document indexing
- **End users**: Public institution staff who need to search and retrieve information from internal documents using natural language queries

### Security Focus

The system is designed with security as a primary concern, operating entirely within the institution's internal network without external dependencies, ensuring sensitive public documents remain secure.