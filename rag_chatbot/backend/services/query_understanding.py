"""
Query Understanding Service.

Analyzes user queries to improve retrieval:
- Detect query intent (title, author, abstract, technical detail, etc.)
- Generate search filters based on intent
- Boost relevant pages/sections
"""
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class QueryIntent:
    """Detected query intent and retrieval hints."""
    intent_type: str  # "title", "author", "abstract", "metadata", "technical", "general"
    confidence: float
    page_filter: Optional[List[int]] = None  # Pages to prioritize
    boost_metadata: Optional[Dict[str, float]] = None  # Metadata fields to boost
    

class QueryUnderstandingService:
    """
    Understands user queries to optimize retrieval.
    
    Key intents:
    - title: "What is the title?" → Boost page 1
    - author: "Who wrote this?" → Boost page 1
    - abstract: "Summary?" → Boost page 1-2
    - metadata: Year, venue, etc. → Boost page 1
    - technical: Specific details → Search all pages
    """
    
    # Query patterns for different intents
    INTENT_PATTERNS = {
        "title": [
            r"what\s+(is|are)\s+the\s+title",
            r"title\s+of\s+(this|the)\s+document",
            r"what\s+(is\s+)?this\s+paper\s+called",
            r"name\s+of\s+(this|the)\s+paper",
        ],
        "author": [
            r"who\s+(wrote|authored|published)",
            r"author[s]?\s+of",
            r"who\s+(is|are)\s+the\s+author",
        ],
        "abstract": [
            r"(summary|abstract|overview)\s+of",
            r"what\s+(is\s+)?this\s+paper\s+about",
            r"main\s+(idea|contribution|topic)",
        ],
        "year": [
            r"when\s+was\s+(this|it)\s+published",
            r"publication\s+(year|date)",
        ],
        "methodology": [
            r"how\s+did\s+they",
            r"what\s+method",
            r"approach\s+used",
        ],
    }
    
    def analyze_query(self, query: str) -> QueryIntent:
        """
        Analyze query to determine intent.
        
        Args:
            query: User question
            
        Returns:
            QueryIntent with detected intent and retrieval hints
        """
        query_lower = query.lower().strip()
        
        # Check each intent pattern
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return self._get_intent_config(intent)
        
        # Default: general query
        return QueryIntent(
            intent_type="general",
            confidence=1.0,
            page_filter=None,
            boost_metadata=None
        )
    
    def _get_intent_config(self, intent: str) -> QueryIntent:
        """Get retrieval configuration for detected intent."""
        
        if intent == "title":
            return QueryIntent(
                intent_type="title",
                confidence=0.95,
                page_filter=[1],  # Title always on page 1
                boost_metadata={"page_number": 10.0}  # Strongly boost page 1
            )
        
        elif intent == "author":
            return QueryIntent(
                intent_type="author",
                confidence=0.95,
                page_filter=[1],
                boost_metadata={"page_number": 10.0}
            )
        
        elif intent == "abstract":
            return QueryIntent(
                intent_type="abstract",
                confidence=0.90,
                page_filter=[1, 2],  # Abstract on page 1-2
                boost_metadata={"page_number": 5.0}
            )
        
        elif intent == "year":
            return QueryIntent(
                intent_type="year",
                confidence=0.95,
                page_filter=[1],
                boost_metadata={"page_number": 10.0}
            )
        
        elif intent == "methodology":
            return QueryIntent(
                intent_type="methodology",
                confidence=0.85,
                page_filter=None,  # Can be anywhere
                boost_metadata=None
            )
        
        return QueryIntent(intent_type=intent, confidence=0.8)
    
    def expand_query(self, query: str, intent: QueryIntent) -> str:
        """
        Expand query based on intent for better matching.
        
        Args:
            query: Original query
            intent: Detected intent
            
        Returns:
            Expanded query string
        """
        expansions = {
            "title": ["title", "paper name", "document title", "heading"],
            "author": ["author", "researcher", "scientist", "wrote", "written by"],
            "abstract": ["abstract", "summary", "overview", "introduction", "contribution"],
            "year": ["year", "published", "publication date", "when"],
        }
        
        if intent.intent_type in expansions:
            expansion_terms = " ".join(expansions[intent.intent_type])
            return f"{query} {expansion_terms}"
        
        return query
