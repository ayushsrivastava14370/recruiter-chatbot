"""Skill taxonomy + deterministic skill extraction.

Skills are matched with alias lists and word-boundary regexes, so every
extracted skill is *literally present* in the source document. This keeps the
match/gap analysis grounded and explainable (no hallucinated skills).
"""
from __future__ import annotations

import re
from functools import lru_cache

# canonical skill -> (category, [aliases])
SKILL_TAXONOMY: dict[str, tuple[str, list[str]]] = {
    # Programming languages
    "Python": ("Programming", ["python", "python3"]),
    "Java": ("Programming", ["java"]),
    "JavaScript": ("Programming", ["javascript", "js", "es6"]),
    "TypeScript": ("Programming", ["typescript"]),
    "C++": ("Programming", ["c++", "cpp"]),
    "C#": ("Programming", ["c#", "csharp"]),
    "Go": ("Programming", ["golang"]),
    "Scala": ("Programming", ["scala"]),
    "Rust": ("Programming", ["rust"]),
    "SQL": ("Data", ["sql", "t-sql", "pl/sql"]),
    "Bash": ("DevOps", ["bash", "shell scripting"]),
    # ML / AI
    "Machine Learning": ("ML/AI", ["machine learning", "ml models", "ml engineering", "ml model", "predictive modelling", "predictive modeling", "anomaly detection", "outlier detection", "recommendation system", "recommendation systems", "recommender system", "recommender systems", "classification models", "churn prediction"]),
    "Deep Learning": ("ML/AI", ["deep learning", "neural networks", "neural network"]),
    "NLP": ("ML/AI", ["nlp", "natural language processing", "text mining", "text classification", "named entity recognition", "sentiment analysis", "information extraction", "text analytics"]),
    "Computer Vision": ("ML/AI", ["computer vision", "image classification", "object detection", "opencv"]),
    "LLMs": ("ML/AI", ["llm", "llms", "large language models", "large language model", "gpt", "gemini", "llama", "openai api", "azure openai", "claude api", "anthropic claude", "generative ai", "genai", "vlm", "vlms", "vision language models"]),
    "AI Agents": ("ML/AI", ["autogen", "multi-agent", "multi agent", "agentic", "ai agents", "llm agents", "semantic kernel", "crewai", "langgraph", "tool calling", "function calling"]),
    "Embeddings": ("ML/AI", ["embeddings", "embedding models", "sentence transformers", "sentence-transformers", "colbert", "vectorization", "text embeddings"]),
    "Semantic Search": ("ML/AI", ["semantic search", "hybrid search", "vector search", "similarity search", "bm25", "information retrieval", "enterprise search"]),
    "Knowledge Graphs": ("ML/AI", ["knowledge graph", "knowledge graphs", "neo4j", "rdf", "graph database"]),
    "RAG": ("ML/AI", ["rag", "retrieval augmented generation", "retrieval-augmented generation"]),
    "Prompt Engineering": ("ML/AI", ["prompt engineering", "prompt design"]),
    "LangChain": ("ML/AI", ["langchain", "llamaindex", "llama index"]),
    "Vector Databases": ("ML/AI", ["vector database", "vector databases", "faiss", "pinecone", "chromadb", "chroma", "weaviate", "milvus", "pgvector", "qdrant", "vector db", "vector store"]),
    "Transformers": ("ML/AI", ["transformers", "hugging face", "huggingface", "bert", "fine-tuning", "fine tuning"]),
    "PyTorch": ("ML/AI", ["pytorch", "torch"]),
    "TensorFlow": ("ML/AI", ["tensorflow", "keras"]),
    "scikit-learn": ("ML/AI", ["scikit-learn", "sklearn", "scikit learn"]),
    "XGBoost": ("ML/AI", ["xgboost", "lightgbm", "catboost"]),
    "MLOps": ("ML/AI", ["mlops", "mlflow", "kubeflow", "model deployment", "model monitoring"]),
    "Statistics": ("Data", ["statistics", "statistical analysis", "hypothesis testing", "a/b testing", "regression analysis"]),
    "Time Series": ("ML/AI", ["time series", "forecasting"]),
    # Data
    "Pandas": ("Data", ["pandas"]),
    "NumPy": ("Data", ["numpy"]),
    "Spark": ("Data", ["spark", "pyspark", "apache spark"]),
    "Airflow": ("Data", ["airflow", "apache airflow"]),
    "Kafka": ("Data", ["kafka", "apache kafka"]),
    "ETL": ("Data", ["etl", "elt", "data pipelines", "data pipeline"]),
    "Data Warehousing": ("Data", ["data warehouse", "data warehousing", "snowflake", "bigquery", "redshift"]),
    "dbt": ("Data", ["dbt"]),
    "Tableau": ("BI", ["tableau"]),
    "Power BI": ("BI", ["power bi", "powerbi"]),
    "Excel": ("BI", ["excel", "advanced excel", "vlookup", "pivot tables"]),
    "Looker": ("BI", ["looker"]),
    "Data Visualization": ("BI", ["data visualization", "data visualisation", "dashboards", "dashboarding", "matplotlib", "seaborn", "plotly", "kibana"]),
    # Databases
    "PostgreSQL": ("Databases", ["postgresql", "postgres"]),
    "MySQL": ("Databases", ["mysql"]),
    "MongoDB": ("Databases", ["mongodb", "mongo"]),
    "Redis": ("Databases", ["redis"]),
    "Elasticsearch": ("Databases", ["elasticsearch", "opensearch"]),
    # Backend / web
    "FastAPI": ("Backend", ["fastapi"]),
    "Django": ("Backend", ["django"]),
    "Flask": ("Backend", ["flask"]),
    "Node.js": ("Backend", ["node.js", "nodejs", "node js", "express.js", "expressjs"]),
    "Spring Boot": ("Backend", ["spring boot", "spring framework"]),
    "REST APIs": ("Backend", ["rest api", "rest apis", "restful"]),
    "GraphQL": ("Backend", ["graphql"]),
    "gRPC": ("Backend", ["grpc"]),
    "Microservices": ("Backend", ["microservices", "microservice"]),
    "React": ("Frontend", ["react", "react.js", "reactjs"]),
    "Streamlit": ("Frontend", ["streamlit", "gradio"]),
    # Cloud / DevOps
    "AWS": ("Cloud", ["aws", "amazon web services", "sagemaker", "ec2", "s3", "lambda"]),
    "GCP": ("Cloud", ["gcp", "google cloud", "vertex ai"]),
    "Azure": ("Cloud", ["azure"]),
    "Docker": ("DevOps", ["docker", "containerization", "containers"]),
    "Kubernetes": ("DevOps", ["kubernetes", "k8s", "eks", "gke"]),
    "Terraform": ("DevOps", ["terraform", "infrastructure as code"]),
    "CI/CD": ("DevOps", ["ci/cd", "cicd", "github actions", "jenkins", "gitlab ci"]),
    "Git": ("DevOps", ["git", "github", "gitlab"]),
    "Linux": ("DevOps", ["linux", "unix"]),
    "Software Testing": ("Engineering", ["unit testing", "pytest", "junit", "test automation", "tdd"]),
    "System Design": ("Engineering", ["system design", "distributed systems", "scalability", "hld", "lld", "high-level design", "low-level design"]),
    "Data Structures & Algorithms": ("Engineering", ["data structures", "algorithms", "dsa"]),
    "OOP": ("Engineering", ["object-oriented programming", "object oriented programming", "oop", "oops"]),
    # Soft skills / business
    "Communication": ("Soft Skills", ["communication", "presentation skills", "storytelling"]),
    "Stakeholder Management": ("Soft Skills", ["stakeholder management", "stakeholders", "cross-functional"]),
    "Leadership": ("Soft Skills", ["leadership", "led a team", "team lead", "mentoring", "mentored"]),
    "Agile": ("Soft Skills", ["agile", "scrum", "kanban", "jira"]),
    "Problem Solving": ("Soft Skills", ["problem solving", "problem-solving", "analytical thinking"]),
}


def _alias_pattern(alias: str) -> str:
    # Word boundaries that also work for tokens like "c++", "c#", "node.js", "ci/cd"
    return r"(?<![A-Za-z0-9+#])" + re.escape(alias) + r"(?![A-Za-z0-9+#])"


@lru_cache(maxsize=1)
def _compiled() -> list[tuple[str, re.Pattern]]:
    compiled = []
    for canonical, (_, aliases) in SKILL_TAXONOMY.items():
        all_aliases = sorted({canonical.lower(), *aliases}, key=len, reverse=True)
        pattern = "|".join(_alias_pattern(a) for a in all_aliases)
        compiled.append((canonical, re.compile(pattern, re.IGNORECASE)))
    return compiled


# A skill mention is ignored if a negation appears shortly before it in the same sentence,
# e.g. "no experience with Docker", "not familiar with Kubernetes", "without AWS".
_NEGATION = re.compile(
    r"\b(no|not|without|lack(?:s|ing)?(?: of)?|never (?:used|worked)|unfamiliar with|zero)\b(?:(?!\b(?:but|however|though|although|whereas)\b)[^.;\n]){0,40}$",
    re.IGNORECASE,
)


def _is_negated(text: str, start: int) -> bool:
    window = text[max(0, start - 60) : start]
    return bool(_NEGATION.search(window))


def extract_skills(text: str) -> list[str]:
    """Return canonical skills that literally appear in `text` (ordered by taxonomy).

    A skill counts if at least one mention is NOT negated.
    """
    found = []
    for canonical, pattern in _compiled():
        if any(not _is_negated(text, m.start()) for m in pattern.finditer(text)):
            found.append(canonical)
    return found


def normalize_skill(skill: str) -> str:
    """Map a free-text skill (e.g. from an LLM) to its canonical name if known."""
    s = skill.strip()
    for canonical, pattern in _compiled():
        m = pattern.fullmatch(s) or (pattern.search(s) if len(s) <= len(canonical) + 12 else None)
        if m:
            return canonical
    return s


def skill_in_text(skill: str, text: str) -> bool:
    """True if a (possibly non-taxonomy) skill is mentioned in text."""
    for canonical, pattern in _compiled():
        if canonical == skill:
            return any(not _is_negated(text, m.start()) for m in pattern.finditer(text))
    return any(not _is_negated(text, m.start()) for m in re.finditer(_alias_pattern(skill.lower()), text, re.IGNORECASE))


def category_of(skill: str) -> str:
    return SKILL_TAXONOMY.get(skill, ("Other", []))[0]


# Skills that are reasonably IMPLIED by other, more specific skills. Implied skills are credited at a
# reduced rate (see matcher.IMPLIED_CREDIT) and shown with an "implied" label, so the recruiter can see
# the reasoning (e.g. someone building RAG systems is doing NLP even if "NLP" is never written).
SKILL_IMPLICATIONS: dict[str, list[str]] = {
    "RAG": ["LLMs", "NLP", "Semantic Search"],
    "LLMs": ["NLP"],
    "AI Agents": ["LLMs"],
    "Transformers": ["NLP", "Deep Learning"],
    "PyTorch": ["Deep Learning"],
    "TensorFlow": ["Deep Learning"],
    "Deep Learning": ["Machine Learning"],
    "Computer Vision": ["Machine Learning"],
    "NLP": ["Machine Learning"],
    "scikit-learn": ["Machine Learning"],
    "XGBoost": ["Machine Learning"],
    "Vector Databases": ["Embeddings"],
    "FastAPI": ["REST APIs"],
    "Flask": ["REST APIs"],
    "Django": ["REST APIs"],
    "Spring Boot": ["REST APIs"],
    "Node.js": ["JavaScript"],
    "PostgreSQL": ["SQL"],
    "MySQL": ["SQL"],
    "Kubernetes": ["Docker"],
}


def implied_skills(skills: list[str]) -> list[str]:
    """Skills implied (transitively) by the given ones, excluding those already present."""
    have, out, stack = set(skills), [], list(skills)
    while stack:
        for imp in SKILL_IMPLICATIONS.get(stack.pop(), []):
            if imp not in have:
                have.add(imp)
                out.append(imp)
                stack.append(imp)
    return out
