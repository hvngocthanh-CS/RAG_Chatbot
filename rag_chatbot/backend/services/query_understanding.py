"""
Query Understanding Service.

Analyzes user queries to improve retrieval for internal English company documents:
- Detect query intent (policy, procedure, contact, deadline, requirement, definition, summary)
- Generate retrieval hints based on intent
- Boost relevant pages/sections
"""
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class QueryIntent:
    """Detected query intent and retrieval hints."""
    intent_type: str  # "policy", "procedure", "contact", "deadline", "requirement", "definition", "summary", "general"
    confidence: float
    page_filter: Optional[List[int]] = None  # Pages to prioritize
    boost_metadata: Optional[Dict[str, float]] = None  # Metadata fields to boost


class QueryUnderstandingService:
    """
    Understands user queries to optimize retrieval for internal English documents.

    Key intents mapped to internal document use cases:
    - policy:      "What is the policy on X?" → boost page 1-2
    - procedure:   "How do I submit a request?" → full search
    - contact:     "Who should I contact for X?" → boost page 1
    - deadline:    "When is the deadline for X?" → full search
    - requirement: "What documents do I need for X?" → full search
    - definition:  "What is X?" → boost page 1-2
    - summary:     "Give me an overview of X" → boost page 1-2
    """

    # Query patterns for internal English document intents
    INTENT_PATTERNS = {
        "policy": [
            r"what\s+is\s+the\s+policy",
            r"policy\s+(for|on|about|regarding|related\s+to)",
            r"are\s+(we|employees?|staff)\s+allowed\s+to",
            r"is\s+it\s+(allowed|permitted|acceptable|mandatory|required)",
            r"company\s+rule[s]?",
            r"what\s+are\s+the\s+rule[s]?",
            r"code\s+of\s+conduct",
            r"compliance\s+(rule|requirement|policy)",
            r"regulation[s]?\s+(on|for|about)",
            r"guideline[s]?\s+(on|for|about)",
        ],
        "procedure": [
            r"how\s+(do|to|should|can|must)\s+I",
            r"how\s+to\s+\w+",
            r"what\s+are\s+the\s+steps\s+(to|for|in)",
            r"step[s]?\s+(to|for|in|of)",
            r"process\s+(for|of|to)",
            r"procedure\s+(for|to|of)",
            r"how\s+does\s+.+\s+work",
            r"walkthrough\s+(for|of)",
            r"instructions?\s+(for|to|on)",
            r"guide\s+(for|to|on)",
        ],
        "contact": [
            r"who\s+(should|do|can|to)\s+(I\s+)?(contact|call|email|reach|ask|report\s+to)",
            r"who\s+is\s+(responsible|in\s+charge|the\s+owner|the\s+poc)",
            r"point\s+of\s+contact",
            r"who\s+(handles|manages|owns|approves)",
            r"(email|phone|contact)\s+(of|for|address\s+of)",
            r"which\s+(team|department|person)\s+(handles|is\s+responsible)",
        ],
        "deadline": [
            r"when\s+(is|are|was)\s+(the\s+)?(deadline|due\s+date|submission\s+date|cutoff)",
            r"deadline\s+(for|of|to)",
            r"due\s+(date|by)\s+(for|of)?",
            r"by\s+when\s+(should|must|do|is)",
            r"timeline\s+(for|of)",
            r"how\s+long\s+(does|will|should)\s+it\s+take",
            r"turnaround\s+time",
            r"processing\s+time",
            r"when\s+will\s+I\s+(receive|get|hear)",
        ],
        "requirement": [
            r"what\s+(do|should|must)\s+I\s+(need|prepare|provide|submit|bring)",
            r"requirement[s]?\s+(for|to|of)",
            r"prerequisite[s]?\s+(for|to)",
            r"document[s]?\s+(needed|required|to\s+submit|to\s+provide)",
            r"eligib(le|ility)\s+(for|to)",
            r"qualif(y|ication[s]?)\s+(for|to)",
            r"criteria\s+(for|to|of)",
            r"condition[s]?\s+(for|to\s+be)",
        ],
        "definition": [
            r"what\s+(is|are)\s+(a\s+|an\s+|the\s+)?\w+",
            r"define\s+\w+",
            r"definition\s+of",
            r"meaning\s+of",
            r"explain\s+what\s+.+\s+(is|means)",
            r"what\s+does\s+.+\s+mean",
            r"what\s+does\s+.+\s+stand\s+for",
        ],
        "summary": [
            r"(summary|overview|introduction)\s+(of|to|about)",
            r"what\s+is\s+this\s+document\s+about",
            r"(briefly|shortly)\s+(describe|explain|summarize|outline)",
            r"give\s+(me\s+)?(a\s+)?(brief|quick|short|high.level)\s+",
            r"main\s+(point[s]?|topic[s]?|content[s]?|section[s]?)",
            r"summarize\s+(this|the)\s+(document|file|report|policy)",
            r"table\s+of\s+contents",
        ],
        # Intent for numeric/tabular data questions
        "numeric": [
            r"how\s+(many|much|long)",
            r"(probation|trial)\s+(period|time|duration)",
            r"\d+\s+(days?|weeks?|months?|years?)",
            r"(duration|length|time|period)\s+(of|for)",
            r"(what|how)\s+(is|are)\s+the\s+(time|duration|period|number|amount)",
            r"according\s+to\s+(the\s+)?table",
            r"as\s+(shown|stated|specified)\s+in\s+(the\s+)?table",
        ],
        
        # Multi-context questions requiring multi-hop reasoning
        "multi_context": [
            r"new\s+hire",
            r"during\s+(their|my|the)\s+(first|second|third)\s+month",
            r"probation\s+(period|employee)",
            r"trial\s+(period|employee)",
            r"during\s+probation",
            r"while\s+on\s+probation",
        ],
    }

    def analyze_query(self, query: str) -> QueryIntent:
        """
        Analyze query to determine intent.

        Args:
            query: User question (English)

        Returns:
            QueryIntent with detected intent and retrieval hints
        """
        query_lower = query.lower().strip()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return self._get_intent_config(intent)

        return QueryIntent(
            intent_type="general",
            confidence=1.0,
            page_filter=None,
            boost_metadata=None
        )

    def _get_intent_config(self, intent: str) -> QueryIntent:
        """Get retrieval configuration for detected intent.

        Design principles:
        - Never hard-code page filters — document layout varies widely.
        - Keep boost factors moderate (1.5–3x) to avoid burying relevant
          chunks that happen to be on unexpected pages.
        - Table boost is the strongest (3x) because chunk_type metadata is
          reliable, unlike page numbers.
        """
        # Mapping: intent → (confidence, boost_metadata)
        configs = {
            "policy":       (0.92, None),
            "procedure":    (0.90, None),
            "contact":      (0.92, None),
            "deadline":     (0.88, None),
            "requirement":  (0.88, None),
            "definition":   (0.85, None),
            "summary":      (0.90, None),
            "numeric":      (0.90, {"table": 3.0}),
            "multi_context": (0.85, None),
        }

        if intent in configs:
            confidence, boost = configs[intent]
            return QueryIntent(
                intent_type=intent,
                confidence=confidence,
                page_filter=None,
                boost_metadata=boost,
            )

        return QueryIntent(intent_type=intent, confidence=0.80)

    def expand_query(self, query: str, intent: QueryIntent) -> str:
        """
        Expand query based on intent for better semantic matching.

        Args:
            query: Original query
            intent: Detected intent

        Returns:
            Expanded query string
        """
        expansions = {
            "policy": ["policy", "rule", "guideline", "regulation", "compliance", "allowed", "prohibited"],
            "procedure": ["steps", "process", "procedure", "how to", "instructions", "workflow"],
            "contact": ["contact", "responsible", "owner", "email", "team", "department", "point of contact"],
            "deadline": ["deadline", "due date", "timeline", "schedule", "by when", "submission date"],
            "requirement": ["requirement", "needed", "must provide", "eligibility", "criteria", "submit"],
            "definition": ["definition", "meaning", "refers to", "is defined as", "stands for"],
            "summary": ["summary", "overview", "purpose", "scope", "introduction", "about"],
            "numeric": ["table", "duration", "period", "days", "weeks", "months", "time", "number", "amount"],
        }

        if intent.intent_type in expansions:
            expansion_terms = " ".join(expansions[intent.intent_type])
            return f"{query} {expansion_terms}"

        return query
