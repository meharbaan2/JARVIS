import math
import re
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from embedding_service import DEFAULT_HASH_DIMENSIONS, HashEmbeddingProvider


TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.-]+")
EMBEDDING_DIMENSIONS = DEFAULT_HASH_DIMENSIONS
CHUNK_MAX_CHARS = 1400
CHUNK_OVERLAP_CHARS = 180
DEFAULT_EMBEDDING_PROVIDER = HashEmbeddingProvider(EMBEDDING_DIMENSIONS)
ROOT_DOCUMENT_NAMES = {
    "README",
    "README.md",
    "README.rst",
    "readme.md",
    "ARCHITECTURE.md",
    "architecture.md",
    "CHANGELOG.md",
    "CONTRACT.md",
    "TOOL_CONTRACT.md",
    "pyproject.toml",
    "package.json",
    "CMakeLists.txt",
}
ROOT_DOCUMENT_SUFFIXES = {
    ".csproj",
    ".fsproj",
    ".vbproj",
    ".sln",
    ".vcxproj",
    ".props",
    ".targets",
    ".proto",
}
DOCS_SUFFIXES = {".md", ".rst", ".txt"}
PROJECT_DOCUMENT_IGNORED_DIRS = {
    ".git",
    ".vs",
    "__pycache__",
    "bin",
    "build",
    "dist",
    "node_modules",
    "obj",
    "runtime_state",
    "venv",
    ".venv",
}
SOURCE_SUMMARY_SUFFIXES = {
    ".py",
    ".cs",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".hpp",
    ".hh",
    ".h",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".proto",
}
SOURCE_SUMMARY_IGNORED_DIRS = {
    ".git",
    ".vs",
    "__pycache__",
    "bin",
    "build",
    "dist",
    "node_modules",
    "obj",
    "runtime_state",
    "venv",
    ".venv",
}


def _tokens(text):
    return [token.lower() for token in TOKEN_RE.findall(text or "") if len(token) > 1]


QUERY_SYNONYMS = {
    "risk": {"security", "threat", "suspicious", "malicious", "verdict"},
    "malware": {"malicious", "suspicious", "threat", "security"},
    "malicious": {"malware", "suspicious", "threat", "security"},
    "safe": {"risk", "security", "verdict"},
    "security": {"risk", "threat", "malicious", "audit"},
    "download": {"internet", "zone", "motw", "security", "risk"},
    "archive": {"zip", "contents", "executable", "script"},
    "quarantine": {"security", "isolate", "restore", "risk"},
    "defender": {"antivirus", "scan", "security", "windows"},
    "probability": {"confidence", "bayesian", "evidence"},
    "confidence": {"probability", "bayesian", "evidence", "reliable"},
    "reliability": {"confidence", "probability", "evidence"},
    "artifact": {"result", "metrics", "output"},
    "scan": {"assessment", "security", "audit"},
    "audit": {"review", "security", "dependencies", "secrets"},
    "ofdm": {"ofdmsim", "ber", "snr", "simulation"},
    "ofdmsim": {"ofdm", "ber", "snr", "simulation"},
    "cad": {"cadconverter", "glb", "step", "geometry"},
    "cadconverter": {"cad", "glb", "step", "geometry"},
    "folderguardian": {"encrypt", "decrypt", "watch", "logs", "protection"},
}


def _expanded_query_tokens(query):
    tokens = Counter(_tokens(query))
    for token in list(tokens):
        for synonym in QUERY_SYNONYMS.get(token, set()):
            tokens[synonym] += 0.35
    return tokens


def _source_kind(project, source_path):
    project = str(project or "")
    source_path = str(source_path or "")
    if project.startswith("tool:") or source_path.startswith("tool_runs/"):
        return "tool_run"
    if source_path == "__source_summary__.md":
        return "source_summary"
    if source_path == "__project_overview__.md":
        return "project_overview"
    suffix = Path(source_path).suffix.lower()
    if suffix in SOURCE_SUMMARY_SUFFIXES:
        return "source"
    return "document"


def _source_kind_bonus(kind, query_tokens):
    tokens = set(query_tokens)
    if kind == "tool_run" and tokens & {"run", "result", "results", "metric", "metrics", "artifact", "ber", "probability", "confidence", "reliability", "risk", "security", "scan", "defender", "quarantine", "archive"}:
        return 1.0
    if kind == "source_summary" and tokens & {"source", "function", "class", "method", "api", "service", "adapter"}:
        return 0.8
    if kind == "project_overview" and tokens & {"overview", "files", "project", "structure", "modules"}:
        return 0.7
    if kind == "document" and tokens & {"security", "audit", "risk", "dependencies", "secrets"}:
        return 0.3
    return 0.0


def _diversify_hits(hits, limit=5, max_per_source=2):
    limit = max(1, int(limit or 5))
    max_per_source = max(1, int(max_per_source or 2))
    selected = []
    per_source = Counter()
    deferred = []
    for hit in hits:
        key = (hit.get("project"), hit.get("source_path"))
        if per_source[key] < max_per_source:
            selected.append(hit)
            per_source[key] += 1
        else:
            deferred.append(hit)
        if len(selected) >= limit:
            return selected
    for hit in deferred:
        selected.append(hit)
        if len(selected) >= limit:
            break
    return selected


def _project_document_candidates(root):
    root = Path(root)
    candidates = []
    seen = set()

    def add(path):
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(path)

    if root.exists():
        for path in sorted(root.iterdir()):
            if not path.is_file():
                continue
            if path.name in ROOT_DOCUMENT_NAMES or path.suffix.lower() in ROOT_DOCUMENT_SUFFIXES:
                add(path)

    docs_dir = root / "docs"
    if docs_dir.exists():
        for path in sorted(path for path in docs_dir.rglob("*") if path.is_file() and path.suffix.lower() in DOCS_SUFFIXES):
            add(path)

    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative_parts = set(path.relative_to(root).parts[:-1])
            if relative_parts & PROJECT_DOCUMENT_IGNORED_DIRS:
                continue
            if path.suffix.lower() in DOCS_SUFFIXES or path.name in ROOT_DOCUMENT_NAMES or path.suffix.lower() in ROOT_DOCUMENT_SUFFIXES:
                add(path)

    return candidates


def _source_summary_candidates(root, max_files=80):
    root = Path(root)
    if not root.exists():
        return []
    candidates = []
    for path in sorted(root.rglob("*")):
        if len(candidates) >= max_files:
            break
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUMMARY_SUFFIXES:
            continue
        relative_parts = set(path.relative_to(root).parts[:-1])
        if relative_parts & SOURCE_SUMMARY_IGNORED_DIRS:
            continue
        candidates.append(path)
    return candidates


def _source_symbols(path, content, limit=12):
    suffix = Path(path).suffix.lower()
    patterns = []
    if suffix == ".py":
        patterns = [
            ("class", r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("function", r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("async function", r"^\s*async\s+def\s+([A-Za-z_][A-Za-z0-9_]*)"),
        ]
    elif suffix == ".cs":
        patterns = [
            ("class", r"\b(?:class|record|struct|interface)\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("method", r"\b(?:public|private|internal|protected|static|async|\s)+(?:[A-Za-z_][A-Za-z0-9_<>,\[\]?]+\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
        ]
    elif suffix in {".cpp", ".cc", ".cxx", ".c", ".hpp", ".hh", ".h"}:
        patterns = [
            ("class", r"\b(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("function", r"\b(?:[A-Za-z_][A-Za-z0-9_:<>*&\s]+)\s+([A-Za-z_][A-Za-z0-9_:]*)\s*\([^;{}]*\)\s*(?:const\s*)?\{"),
        ]
    elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
        patterns = [
            ("class", r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("function", r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("function", r"\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s*)?\("),
        ]
    elif suffix == ".proto":
        patterns = [
            ("service", r"\bservice\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("message", r"\bmessage\s+([A-Za-z_][A-Za-z0-9_]*)"),
            ("rpc", r"\brpc\s+([A-Za-z_][A-Za-z0-9_]*)"),
        ]

    symbols = []
    seen = set()
    for kind, pattern in patterns:
        for match in re.finditer(pattern, content, flags=re.MULTILINE):
            symbol = match.group(1)
            key = (kind, symbol)
            if key in seen:
                continue
            seen.add(key)
            symbols.append(f"{kind} {symbol}")
            if len(symbols) >= limit:
                return symbols
    return symbols


def _build_source_summary(project, root, max_files=80, max_bytes_per_file=64_000):
    root = Path(root)
    candidates = _source_summary_candidates(root, max_files=max_files)
    if not candidates:
        return ""

    extension_counts = Counter(path.suffix.lower() for path in candidates)
    lines = [f"# {project} Source Summary", "", "## File Counts"]
    for suffix, count in sorted(extension_counts.items()):
        lines.append(f"- {suffix}: {count}")

    lines.extend(["", "## Source Files"])
    for path in candidates:
        relative = path.relative_to(root)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")[:max_bytes_per_file]
        except OSError:
            content = ""
        symbols = _source_symbols(path, content)
        symbol_text = f" symbols: {', '.join(symbols)}" if symbols else ""
        lines.append(f"- {relative}{symbol_text}")

    return "\n".join(lines)


def _build_project_overview(project, root, max_files=250):
    root = Path(root)
    if not root.exists():
        return ""
    files = []
    ignored = PROJECT_DOCUMENT_IGNORED_DIRS | SOURCE_SUMMARY_IGNORED_DIRS
    for path in sorted(root.rglob("*")):
        if len(files) >= max_files:
            break
        if not path.is_file():
            continue
        relative_parts = set(path.relative_to(root).parts[:-1])
        if relative_parts & ignored:
            continue
        files.append(path)
    if not files:
        return ""
    extension_counts = Counter(path.suffix.lower() or "<none>" for path in files)
    manifest_files = [
        path.relative_to(root)
        for path in files
        if path.name in ROOT_DOCUMENT_NAMES or path.suffix.lower() in ROOT_DOCUMENT_SUFFIXES
    ]
    source_files = [path.relative_to(root) for path in files if path.suffix.lower() in SOURCE_SUMMARY_SUFFIXES]
    lines = [
        f"# {project} Project Overview",
        "",
        f"Root: {root}",
        f"Indexed file sample count: {len(files)}",
        "",
        "## File Types",
    ]
    for suffix, count in sorted(extension_counts.items()):
        lines.append(f"- {suffix}: {count}")
    if manifest_files:
        lines.extend(["", "## Manifests and Docs"])
        lines.extend(f"- {path}" for path in manifest_files[:40])
    if source_files:
        lines.extend(["", "## Source File Sample"])
        lines.extend(f"- {path}" for path in source_files[:80])
    return "\n".join(lines)


def _artifact_memory_text(tool_name, path, data):
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    sections = [
        f"Tool artifact: {tool_name}.{data.get('capability', 'unknown')}",
        f"Artifact path: {path}",
        f"Status: {data.get('status', 'unknown')}",
        f"Created: {data.get('created_at') or data.get('timestamp') or 'unknown'}",
    ]
    for key in (
        "target_path",
        "risk_level",
        "risk_score",
        "verdict",
        "evidence_status",
    ):
        value = data.get(key) if data.get(key) is not None else metrics.get(key)
        if value is not None:
            sections.append(f"{key}: {value}")
    compact_metrics = {}
    for key in (
        "ber",
        "errors",
        "bits",
        "baseline_ber",
        "emi_ber",
        "risk_score",
        "risk_level",
        "manifest_count",
        "scanned_text_file_count",
        "validated_count",
        "failed_count",
        "processed_count",
    ):
        if key in metrics:
            compact_metrics[key] = metrics[key]
    if metrics.get("summary") and isinstance(metrics["summary"], dict):
        compact_metrics["summary"] = metrics["summary"]
    if compact_metrics:
        sections.append(f"Metrics: {json.dumps(compact_metrics, sort_keys=True)}")
    findings = data.get("findings") or metrics.get("findings") or []
    if isinstance(findings, list) and findings:
        sections.append("Findings:")
        for finding in findings[:10]:
            if isinstance(finding, dict):
                sections.append(
                    f"- {finding.get('severity', 'info')}: {finding.get('message', 'finding')} "
                    f"({finding.get('path', 'unknown path')})"
                )
            else:
                sections.append(f"- {finding}")
    detectors = data.get("detectors") or metrics.get("detectors") or []
    if isinstance(detectors, list) and detectors:
        sections.append("Detectors:")
        for detector in detectors[:12]:
            if isinstance(detector, dict):
                sections.append(
                    f"- {detector.get('name', 'detector')}: {detector.get('status', 'unknown')} "
                    f"findings={detector.get('finding_count', detector.get('event_count', 0))}"
                )
    recommendations = data.get("recommended_actions") or metrics.get("recommended_actions") or []
    if isinstance(recommendations, list) and recommendations:
        sections.append("Recommended actions:")
        sections.extend(f"- {item}" for item in recommendations[:6])
    bayesian_evidence = data.get("bayesian_evidence") or metrics.get("bayesian_evidence") or []
    if isinstance(bayesian_evidence, list) and bayesian_evidence:
        sections.append("Bayesian evidence:")
        for item in bayesian_evidence[:8]:
            if isinstance(item, dict):
                sections.append(
                    f"- {item.get('status', 'unknown')} {item.get('factor', 'factor')} "
                    f"passed={item.get('passed')} observed={item.get('observed_value')}"
                )
    bayesian_assessment = data.get("bayesian_assessment") or metrics.get("bayesian_assessment") or {}
    if isinstance(bayesian_assessment, dict) and bayesian_assessment:
        sections.append(f"Bayesian assessment: {json.dumps(bayesian_assessment, sort_keys=True)}")
    stdout = data.get("stdout") or ""
    if stdout:
        sections.append(f"Output: {str(stdout)[:3000]}")
    stderr = data.get("stderr") or ""
    if stderr:
        sections.append(f"Error: {str(stderr)[:1500]}")
    return "\n".join(sections)


def _stable_bucket(token, dimensions=EMBEDDING_DIMENSIONS):
    return HashEmbeddingProvider(dimensions)._bucket(token)


def _embedding(text, dimensions=EMBEDDING_DIMENSIONS):
    return HashEmbeddingProvider(dimensions).embed(text)


def _content_hash(text):
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _cosine(left, right):
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _document_chunks(content, max_chars=CHUNK_MAX_CHARS, overlap_chars=CHUNK_OVERLAP_CHARS):
    text = (content or "").replace("\r\n", "\n").strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + max_chars)
                chunks.append(paragraph[start:end].strip())
                if end == len(paragraph):
                    break
                start = max(0, end - overlap_chars)
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())
    return chunks


class MemoryService:
    """Structured memory plus lightweight local retrieval without extra dependencies."""

    def __init__(self, db_path, summary_frequency=3, max_summaries_to_keep=100, embedding_provider=None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.summary_frequency = summary_frequency
        self.max_summaries_to_keep = max_summaries_to_keep
        self.embedding_provider = embedding_provider or DEFAULT_EMBEDDING_PROVIDER
        self.current_session = {
            "mistral": [],
            "wizardmath": [],
            "phi3": [],
            "runtime": [],
        }
        self.init_db()

    def init_db(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                original_count INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS project_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                source_path TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                token_index TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, source_path)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS project_document_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project TEXT NOT NULL,
                source_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                token_index TEXT NOT NULL,
                embedding_json TEXT NOT NULL,
                embedding_provider TEXT,
                embedding_dimensions INTEGER,
                content_hash TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project, source_path, chunk_index)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                capability TEXT NOT NULL,
                input_json TEXT NOT NULL,
                status TEXT NOT NULL,
                output_text TEXT,
                error_text TEXT,
                artifact_path TEXT,
                metrics_json TEXT,
                evidence_status TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                finished_at DATETIME
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS runtime_execution_plans (
                id TEXT PRIMARY KEY,
                intent TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                finished_at TEXT,
                result_summary TEXT,
                error TEXT,
                steps_json TEXT NOT NULL,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._ensure_column("tool_runs", "artifact_path", "TEXT")
        self._ensure_column("tool_runs", "metrics_json", "TEXT")
        self._ensure_column("tool_runs", "evidence_status", "TEXT")
        self._ensure_column("project_document_chunks", "embedding_provider", "TEXT")
        self._ensure_column("project_document_chunks", "embedding_dimensions", "INTEGER")
        self._ensure_column("project_document_chunks", "content_hash", "TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_type ON memory_summaries (model_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_documents_project ON project_documents (project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_project_document_chunks_project ON project_document_chunks (project)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_runs_tool ON tool_runs (tool_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runtime_execution_plans_status ON runtime_execution_plans (status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runtime_execution_plans_created ON runtime_execution_plans (created_at)")
        self._backfill_document_chunks(cursor)
        self.conn.commit()

    def _ensure_column(self, table, column, definition):
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in cursor.fetchall()}
        if column not in columns:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def add_conversation(self, model_type, role, content):
        self.current_session.setdefault(model_type, []).append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_memory (model_type, role, content) VALUES (?, ?, ?)",
            (model_type, role, content),
        )
        self.conn.commit()

    def add_summary(self, model_type, summary, original_count):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memory_summaries (model_type, summary, original_count) VALUES (?, ?, ?)",
            (model_type, summary, original_count),
        )
        self.conn.commit()
        self.current_session[model_type] = []
        self.cleanup_old_summaries(model_type)

    def cleanup_old_summaries(self, model_type):
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memory_summaries WHERE model_type = ?", (model_type,))
        count = cursor.fetchone()[0]
        if count > self.max_summaries_to_keep:
            cursor.execute(
                """
                DELETE FROM memory_summaries
                WHERE id IN (
                    SELECT id FROM memory_summaries
                    WHERE model_type = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
                """,
                (model_type, count - self.max_summaries_to_keep),
            )
            self.conn.commit()

    def recent_summaries(self, model_type, limit=2):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT summary FROM memory_summaries
            WHERE model_type = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (model_type, limit),
        )
        return [row[0] for row in cursor.fetchall()]

    def context_messages(self, model_type, new_query):
        messages = [
            {"role": "system", "content": f"Previous conversation summary: {summary}"}
            for summary in self.recent_summaries(model_type)
        ]
        messages.extend({"role": msg["role"], "content": msg["content"]} for msg in self.current_session.get(model_type, []))
        for hit in self.search(new_query, limit=4):
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"Relevant project memory from {hit['project']} "
                        f"({hit['source_path']} chunk {hit.get('chunk_index', 0)}): {hit['snippet']}"
                    ),
                }
            )
        messages.append({"role": "user", "content": new_query})
        return messages

    def index_document(self, project, source_path, content, title=None):
        token_index = " ".join(_tokens(content))
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO project_documents (project, source_path, title, content, token_index, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(project, source_path) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                token_index = excluded.token_index,
                updated_at = CURRENT_TIMESTAMP
            """,
            (project, str(source_path), title, content, token_index),
        )
        self._replace_document_chunks(cursor, project, str(source_path), content, title)
        self.conn.commit()

    def index_project_documents(self, project, root, max_files=24, max_bytes_per_file=128_000):
        root = Path(root)
        if not root.exists():
            return 0

        indexed = 0
        seen = set()
        for path in _project_document_candidates(root):
            resolved = path.resolve()
            if resolved in seen or indexed >= max_files:
                continue
            seen.add(resolved)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")[:max_bytes_per_file]
            except OSError:
                continue
            self.index_document(project, path, content, title=f"{project} {path.name}")
            indexed += 1
        return indexed

    def index_project_source_summary(self, project, root, max_files=80):
        root = Path(root)
        if not root.exists():
            return 0
        summary = _build_source_summary(project, root, max_files=max_files)
        if not summary:
            return 0
        self.index_document(
            project,
            "__source_summary__.md",
            summary,
            title=f"{project} source summary",
        )
        return 1

    def index_project_overview(self, project, root, max_files=250):
        root = Path(root)
        if not root.exists():
            return 0
        overview = _build_project_overview(project, root, max_files=max_files)
        if not overview:
            return 0
        self.index_document(
            project,
            "__project_overview__.md",
            overview,
            title=f"{project} project overview",
        )
        return 1

    def index_tool_artifacts(self, tool_name, artifact_root, max_files=50, max_bytes_per_file=96_000):
        artifact_root = Path(artifact_root)
        if not artifact_root.exists():
            return 0
        artifacts = sorted(
            (path for path in artifact_root.glob("*.json") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[: max(1, int(max_files or 50))]
        indexed = 0
        for path in artifacts:
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")[:max_bytes_per_file]
                data = json.loads(raw)
            except (OSError, json.JSONDecodeError):
                continue
            content = _artifact_memory_text(tool_name, path, data)
            self.index_document(
                f"tool:{tool_name}",
                f"artifacts/{path.name}",
                content,
                title=f"{tool_name} artifact {path.name}",
            )
            indexed += 1
        return indexed

    def index_readme(self, project, root):
        return self.index_project_documents(project, root) > 0

    def search(self, query, limit=5, project=None, source_kind=None, max_per_source=2):
        query_tokens = _expanded_query_tokens(query)
        if not query_tokens:
            return []
        query_embedding = self.embedding_provider.embed(query)

        cursor = self.conn.cursor()
        if project:
            cursor.execute(
                """
                SELECT project, source_path, title, content, token_index, embedding_json,
                       chunk_index, embedding_provider, embedding_dimensions
                FROM project_document_chunks
                WHERE project = ?
                """,
                (project,),
            )
        else:
            cursor.execute(
                """
                SELECT project, source_path, title, content, token_index, embedding_json,
                       chunk_index, embedding_provider, embedding_dimensions
                FROM project_document_chunks
                """
            )
        rows = cursor.fetchall()
        if source_kind:
            rows = [
                row for row in rows
                if _source_kind(row[0], row[1]) == source_kind
            ]
        document_frequency = Counter()
        for row in rows:
            document_frequency.update(set(str(row[4] or "").split()))
        row_count = max(1, len(rows))
        hits = []
        active_provider = getattr(self.embedding_provider, "name", "hash")
        active_dimensions = len(query_embedding)
        for (
            row_project,
            source_path,
            title,
            content,
            token_index,
            embedding_json,
            chunk_index,
            embedding_provider,
            embedding_dimensions,
        ) in rows:
            doc_tokens = Counter(token_index.split())
            lexical_score = 0.0
            matched_tokens = set()
            for token, query_count in query_tokens.items():
                if token in doc_tokens:
                    matched_tokens.add(token)
                    idf = math.log((1 + row_count) / (1 + document_frequency.get(token, 0))) + 1.0
                    lexical_score += query_count * (1 + math.log(1 + doc_tokens[token])) * idf
            try:
                stored_vector = json.loads(embedding_json)
                same_provider = not embedding_provider or embedding_provider == active_provider
                same_dimensions = not embedding_dimensions or int(embedding_dimensions) == active_dimensions
                vector_score = max(0.0, _cosine(query_embedding, stored_vector)) if same_provider and same_dimensions else 0.0
            except (TypeError, ValueError, json.JSONDecodeError):
                vector_score = 0.0
            phrase_bonus = 2.0 if query.lower() in content.lower() else 0.0
            title_text = f"{title or ''} {source_path}".lower()
            title_bonus = sum(0.75 for token in query_tokens if token in title_text)
            path_bonus = sum(0.45 for token in query_tokens if token in str(source_path).lower())
            project_bonus = 0.75 if row_project.lower() in query.lower() else 0.0
            tool_bonus = 0.5 if row_project.startswith("tool:") and row_project[5:] in query.lower() else 0.0
            coverage_bonus = 1.5 * (len(matched_tokens) / max(1, len(query_tokens)))
            kind = _source_kind(row_project, source_path)
            kind_bonus = _source_kind_bonus(kind, query_tokens)
            score = (
                lexical_score
                + (2.0 * vector_score)
                + phrase_bonus
                + title_bonus
                + path_bonus
                + project_bonus
                + tool_bonus
                + coverage_bonus
                + kind_bonus
            )
            if score <= 0:
                continue
            snippet = self._snippet(content, query_tokens)
            hits.append(
                {
                    "project": row_project,
                    "source_path": source_path,
                    "title": title,
                    "score": score,
                    "lexical_score": lexical_score,
                    "vector_score": vector_score,
                    "phrase_bonus": phrase_bonus,
                    "title_bonus": title_bonus,
                    "path_bonus": path_bonus,
                    "coverage_bonus": coverage_bonus,
                    "matched_tokens": sorted(matched_tokens),
                    "source_kind": kind,
                    "chunk_index": chunk_index,
                    "snippet": snippet,
                }
            )
        hits.sort(key=lambda item: item["score"], reverse=True)
        return _diversify_hits(hits, limit=limit, max_per_source=max_per_source)

    def stats(self):
        cursor = self.conn.cursor()
        counts = {}
        for table in (
            "conversation_memory",
            "memory_summaries",
            "project_documents",
            "project_document_chunks",
            "tool_runs",
            "runtime_execution_plans",
        ):
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        cursor.execute(
            """
            SELECT project, COUNT(*)
            FROM project_documents
            GROUP BY project
            ORDER BY project
            """
        )
        project_counts = {row[0]: row[1] for row in cursor.fetchall()}
        return {
            "database_path": str(self.db_path),
            "conversation_count": counts["conversation_memory"],
            "summary_count": counts["memory_summaries"],
            "project_document_count": counts["project_documents"],
            "project_document_chunk_count": counts["project_document_chunks"],
            "embedding_provider": self.embedding_provider.health().get("name", getattr(self.embedding_provider, "name", "hash")),
            "embedding_dimensions": self.embedding_provider.health().get("dimensions", getattr(self.embedding_provider, "dimensions", EMBEDDING_DIMENSIONS)),
            "tool_run_count": counts["tool_runs"],
            "execution_plan_count": counts["runtime_execution_plans"],
            "project_document_counts": project_counts,
            "active_sessions": {
                model_type: len(messages)
                for model_type, messages in self.current_session.items()
            },
        }

    def recent_tool_runs(self, limit=8):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT tool_name, capability, status, input_json, artifact_path,
                   metrics_json, evidence_status, started_at, finished_at
            FROM tool_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            {
                "tool_name": row[0],
                "capability": row[1],
                "status": row[2],
                "input_json": row[3],
                "artifact_path": row[4],
                "metrics_json": row[5],
                "evidence_status": row[6],
                "started_at": row[7],
                "finished_at": row[8],
            }
            for row in cursor.fetchall()
        ]

    def record_execution_plan(self, plan):
        payload = plan.to_dict() if hasattr(plan, "to_dict") else dict(plan)
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO runtime_execution_plans (
                id, intent, source, status, created_at, finished_at,
                result_summary, error, steps_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                intent = excluded.intent,
                source = excluded.source,
                status = excluded.status,
                created_at = excluded.created_at,
                finished_at = excluded.finished_at,
                result_summary = excluded.result_summary,
                error = excluded.error,
                steps_json = excluded.steps_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                payload.get("id"),
                payload.get("intent", ""),
                payload.get("source", ""),
                payload.get("status", ""),
                payload.get("created_at", ""),
                payload.get("finished_at") or "",
                payload.get("result_summary") or "",
                payload.get("error") or "",
                json.dumps(payload.get("steps", []), sort_keys=True),
            ),
        )
        self.conn.commit()
        return payload.get("id")

    def recent_execution_plans(self, limit=8):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT id, intent, source, status, created_at, finished_at,
                   result_summary, error, steps_json
            FROM runtime_execution_plans
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        plans = []
        for row in cursor.fetchall():
            try:
                steps = json.loads(row[8] or "[]")
            except json.JSONDecodeError:
                steps = []
            plans.append(
                {
                    "id": row[0],
                    "intent": row[1],
                    "source": row[2],
                    "status": row[3],
                    "created_at": row[4],
                    "finished_at": row[5],
                    "result_summary": row[6],
                    "error": row[7],
                    "steps": steps,
                }
            )
        return plans

    def record_tool_run(
        self,
        tool_name,
        capability,
        input_json,
        status,
        output_text="",
        error_text="",
        artifact_path=None,
        metrics_json=None,
        evidence_status=None,
    ):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO tool_runs (
                tool_name, capability, input_json, status, output_text, error_text,
                artifact_path, metrics_json, evidence_status, finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                tool_name,
                capability,
                input_json,
                status,
                output_text,
                error_text,
                str(artifact_path) if artifact_path else None,
                metrics_json,
                evidence_status,
            ),
        )
        self.conn.commit()
        run_id = cursor.lastrowid
        self._index_tool_run_memory(run_id, tool_name, capability, input_json, status, output_text, error_text, metrics_json, artifact_path)
        return run_id

    def _replace_document_chunks(self, cursor, project, source_path, content, title=None):
        cursor.execute(
            "DELETE FROM project_document_chunks WHERE project = ? AND source_path = ?",
            (project, str(source_path)),
        )
        chunks = _document_chunks(content)
        if not chunks and content:
            chunks = [content]
        for index, chunk in enumerate(chunks):
            vector = self.embedding_provider.embed(chunk)
            content_hash = _content_hash(chunk)
            cursor.execute(
                """
                INSERT INTO project_document_chunks (
                    project, source_path, chunk_index, title, content,
                    token_index, embedding_json, embedding_provider,
                    embedding_dimensions, content_hash, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    project,
                    str(source_path),
                    index,
                    title,
                    chunk,
                    " ".join(_tokens(chunk)),
                    json.dumps(vector, separators=(",", ":")),
                    getattr(self.embedding_provider, "name", "hash"),
                    len(vector),
                    content_hash,
                ),
            )

    def _backfill_document_chunks(self, cursor):
        cursor.execute(
            """
            SELECT project, source_path, title, content
            FROM project_documents
            WHERE NOT EXISTS (
                SELECT 1 FROM project_document_chunks
                WHERE project_document_chunks.project = project_documents.project
                  AND project_document_chunks.source_path = project_documents.source_path
            )
            """
        )
        for project, source_path, title, content in cursor.fetchall():
            self._replace_document_chunks(cursor, project, source_path, content, title)

    def _index_tool_run_memory(self, run_id, tool_name, capability, input_json, status, output_text, error_text, metrics_json, artifact_path):
        sections = [
            f"Tool run {run_id}: {tool_name}.{capability}",
            f"Status: {status}",
            f"Input: {input_json}",
        ]
        if metrics_json:
            sections.append(f"Metrics: {metrics_json}")
        if artifact_path:
            sections.append(f"Artifact: {artifact_path}")
        if output_text:
            sections.append(f"Output: {output_text[:6000]}")
        if error_text:
            sections.append(f"Error: {error_text[:3000]}")
        content = "\n".join(sections)
        self.index_document(
            f"tool:{tool_name}",
            f"tool_runs/{run_id}_{capability}",
            content,
            title=f"{tool_name}.{capability} run {run_id}",
        )

    def _snippet(self, content, query_tokens, max_chars=450):
        lowered = content.lower()
        first = min((lowered.find(token) for token in query_tokens if lowered.find(token) >= 0), default=0)
        start = max(0, first - 120)
        end = min(len(content), start + max_chars)
        return content[start:end].replace("\n", " ").strip()
