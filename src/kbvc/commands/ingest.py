# kbvc/commands/ingest.py
"""
kbvc ingest — pull external knowledge sources into the repo.

Design principle:
  kbvc ingest downloads + converts → writes .md files to the repo
  kbvc add     stages those files for embedding
  kbvc commit  embeds and creates the commit object
  kbvc push    pushes vectors to a remote vector DB

This separation means:
  - ingest never touches the vector DB
  - you can inspect the downloaded content before committing
  - ingested files are normal .md files — reviewable, diffable, git-trackable

Supported sources (v1):
  website   → fetches a URL, converts HTML → Markdown
  github    → clones/pulls a repo, imports .md files
  pdf       → converts PDF → Markdown (requires pypdf2 or pdfminer)
  notion    → imports from Notion via their export API (requires notion-client)
  text      → imports a raw text file with frontmatter generation
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# IngestResult
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    """Returned by every ingest handler. Always writes one or more .md files."""
    source_url: str
    output_files: list[Path]
    bytes_fetched: int
    warnings: list[str]


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------

def _slugify(text: str, max_len: int = 60) -> str:
    """Convert arbitrary text to a safe filename slug."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:max_len].rstrip("-") or "untitled"


def _make_frontmatter(
    ko_id: str,
    source_url: str,
    title: str,
    source_type: str = "web",
    tags: Optional[list] = None,
) -> str:
    tag_str = str(tags or [source_type])
    return (
        f"---\n"
        f"id: {ko_id}\n"
        f"type: document\n"
        f"source_type: {source_type}\n"
        f"source_url: {source_url}\n"
        f"title: {title!r}\n"
        f"tags: {tag_str}\n"
        f"volatility: slow\n"
        f"ingested_at: {date.today().isoformat()!r}\n"
        f"---\n\n"
    )


# ---------------------------------------------------------------------------
# Website ingest
# ---------------------------------------------------------------------------

def ingest_website(
    url: str,
    output_dir: Path,
    force: bool = False,
) -> IngestResult:
    """
    Fetch a web page and convert it to Markdown.

    Uses the standard library urllib (no external deps in v1).
    For JavaScript-heavy pages, recommend the user pre-fetch with wget/curl.
    """
    import urllib.request

    parsed = urlparse(url)
    slug = _slugify(parsed.netloc.replace(".", "-") + "-" + parsed.path)
    out_path = output_dir / f"{slug}.md"

    if out_path.exists() and not force:
        return IngestResult(url, [out_path], 0,
                            [f"File already exists (use --force to overwrite): {out_path}"])

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "KBVC-Ingest/0.1 (+https://github.com/kbvc/kbvc)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html_bytes = resp.read()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {url}: {exc}")

    html = html_bytes.decode("utf-8", errors="replace")
    markdown, title = _html_to_markdown(html, url)
    ko_id = _slugify(title or slug)

    content = _make_frontmatter(ko_id, url, title or url, source_type="web") + markdown
    out_path.write_text(content, encoding="utf-8")

    return IngestResult(url, [out_path], len(html_bytes), [])


def _html_to_markdown(html: str, url: str) -> tuple[str, str]:
    """
    Minimal HTML → Markdown converter using stdlib re (no beautifulsoup).
    Sufficient for documentation pages; complex JS apps need pre-fetching.
    """
    # Extract title
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = re.sub(r"\s+", " ", title_m.group(1).strip()) if title_m else ""

    # Strip scripts, styles, nav, footer
    html = re.sub(r"<(script|style|nav|footer|header|aside)[^>]*>.*?</\1>",
                  "", html, flags=re.IGNORECASE | re.DOTALL)

    # Convert headings
    for level in [6, 5, 4, 3, 2, 1]:
        html = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            lambda m, lv=level: "\n" + "#" * lv + " " + re.sub(r"<[^>]+>", "", m.group(1)).strip() + "\n",
            html, flags=re.IGNORECASE | re.DOTALL,
        )

    # Convert paragraphs and line breaks
    html = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\1\n", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", html, flags=re.IGNORECASE | re.DOTALL)

    # Convert code blocks
    html = re.sub(r"<pre[^>]*><code[^>]*>(.*?)</code></pre>",
                  lambda m: "\n```\n" + re.sub(r"<[^>]+>", "", m.group(1)) + "\n```\n",
                  html, flags=re.IGNORECASE | re.DOTALL)

    # Convert bold/italic
    html = re.sub(r"<(b|strong)[^>]*>(.*?)</\1>", r"**\2**", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<(i|em)[^>]*>(.*?)</\1>", r"*\2*", html, flags=re.IGNORECASE | re.DOTALL)

    # Convert links
    html = re.sub(r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
                  r"[\2](\1)", html, flags=re.IGNORECASE | re.DOTALL)

    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", html)

    # Decode HTML entities
    entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">",
                "&quot;": '"', "&apos;": "'", "&nbsp;": " "}
    for ent, ch in entities.items():
        text = text.replace(ent, ch)

    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip(), title


# ---------------------------------------------------------------------------
# GitHub ingest
# ---------------------------------------------------------------------------

def ingest_github(
    repo_url: str,
    output_dir: Path,
    branch: str = "main",
    path_filter: str = "**/*.md",
    force: bool = False,
) -> IngestResult:
    """
    Clone or pull a GitHub repo and import Markdown files.

    Files are imported into output_dir/<repo_name>/.
    The frontmatter is enriched with the source GitHub URL.
    """
    import subprocess
    import tempfile

    parsed = urlparse(repo_url)
    repo_name = _slugify(parsed.path.strip("/").replace("/", "-"))
    repo_out = output_dir / repo_name
    repo_out.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []
    warnings: list[str] = []
    total_bytes = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        clone_dir = Path(tmpdir) / "repo"

        # git clone --depth=1 for speed
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", "--branch", branch, repo_url, str(clone_dir)],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"git clone failed for {repo_url}: {exc.stderr.decode()}"
            )
        except FileNotFoundError:
            raise RuntimeError("git not found on PATH.")

        # Collect matching files
        md_files = sorted(clone_dir.rglob("*.md"))
        if not md_files:
            warnings.append(f"No .md files found in {repo_url} (branch: {branch})")

        for src in md_files:
            rel = src.relative_to(clone_dir)
            out = repo_out / rel
            out.parent.mkdir(parents=True, exist_ok=True)

            if out.exists() and not force:
                warnings.append(f"Skipped (exists): {out}")
                output_files.append(out)
                continue

            content = src.read_text(encoding="utf-8", errors="replace")
            total_bytes += len(content)

            # Inject or augment frontmatter
            if not content.startswith("---"):
                ko_id = _slugify(src.stem)
                gh_url = f"{repo_url}/blob/{branch}/{str(rel)}"
                content = _make_frontmatter(ko_id, gh_url, src.stem, source_type="github") + content

            out.write_text(content, encoding="utf-8")
            output_files.append(out)

    return IngestResult(repo_url, output_files, total_bytes, warnings)


# ---------------------------------------------------------------------------
# PDF ingest
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: Path,
    output_dir: Path,
    force: bool = False,
) -> IngestResult:
    """
    Convert a PDF to Markdown using pypdf (optional dependency).
    Falls back to a stub message if pypdf is not installed.
    """
    out_path = output_dir / f"{_slugify(pdf_path.stem)}.md"
    warnings: list[str] = []

    if out_path.exists() and not force:
        return IngestResult(str(pdf_path), [out_path], 0,
                            [f"File exists (use --force): {out_path}"])

    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(
            "pypdf not installed.\n"
            "Run: pip install pypdf\n\n"
            "Alternatively, convert the PDF to text first:\n"
            "  pdftotext input.pdf output.txt"
        )

    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"## Page {i}\n\n{text.strip()}")

    markdown = "\n\n".join(pages)
    ko_id = _slugify(pdf_path.stem)
    title = pdf_path.stem.replace("-", " ").replace("_", " ").title()
    content = _make_frontmatter(ko_id, str(pdf_path), title, source_type="pdf") + markdown

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")

    return IngestResult(str(pdf_path), [out_path], pdf_path.stat().st_size, warnings)


# ---------------------------------------------------------------------------
# Notion ingest
# ---------------------------------------------------------------------------

def ingest_notion(
    page_id: str,
    output_dir: Path,
    notion_token: str,
    force: bool = False,
) -> IngestResult:
    """
    Import a Notion page (and its children) via the Notion API.
    Requires: pip install notion-client
    """
    try:
        from notion_client import Client
    except ImportError:
        raise RuntimeError(
            "notion-client not installed.\n"
            "Run: pip install notion-client"
        )

    client = Client(auth=notion_token)
    output_files: list[Path] = []
    warnings: list[str] = []

    def fetch_page(pid: str, depth: int = 0) -> Optional[Path]:
        try:
            page = client.pages.retrieve(pid)
            title = _extract_notion_title(page)
            slug = _slugify(title or pid[:8])
            out = output_dir / f"{slug}.md"

            if out.exists() and not force:
                warnings.append(f"Skipped (exists): {out}")
                output_files.append(out)
                return out

            blocks = client.blocks.children.list(pid)
            md = _notion_blocks_to_md(blocks.get("results", []))
            source_url = f"https://notion.so/{pid.replace('-', '')}"

            content = _make_frontmatter(slug, source_url, title or slug, "notion") + md
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(content, encoding="utf-8")
            output_files.append(out)
            return out
        except Exception as exc:
            warnings.append(f"Failed to fetch page {pid}: {exc}")
            return None

    fetch_page(page_id)
    return IngestResult(f"notion:{page_id}", output_files,
                        sum(f.stat().st_size for f in output_files if f.exists()), warnings)


def _extract_notion_title(page: dict) -> str:
    try:
        props = page.get("properties", {})
        for key in ["Name", "Title", "title"]:
            if key in props:
                rich = props[key].get("title", [])
                return "".join(r.get("plain_text", "") for r in rich)
    except Exception:
        pass
    return ""


def _notion_blocks_to_md(blocks: list) -> str:
    lines = []
    for block in blocks:
        btype = block.get("type", "")
        data = block.get(btype, {})
        rich = data.get("rich_text", [])
        text = "".join(r.get("plain_text", "") for r in rich)

        if btype == "heading_1":
            lines.append(f"# {text}")
        elif btype == "heading_2":
            lines.append(f"## {text}")
        elif btype == "heading_3":
            lines.append(f"### {text}")
        elif btype in ("paragraph", "quote"):
            lines.append(text or "")
        elif btype == "bulleted_list_item":
            lines.append(f"- {text}")
        elif btype == "numbered_list_item":
            lines.append(f"1. {text}")
        elif btype == "code":
            lang = data.get("language", "")
            lines.append(f"```{lang}\n{text}\n```")
        elif btype == "divider":
            lines.append("---")
        else:
            if text:
                lines.append(text)

    return "\n\n".join(l for l in lines if l.strip())


@dataclass
class IngestTextResult:
    ko_id: str
    output_file: Path


def ingest_text(
    file_path: Path,
    output_dir: Path,
    ko_id: Optional[str] = None,
    ko_type: str = "document",
    force: bool = False,
) -> IngestTextResult:
    """
    Ingest a plain text or Markdown file (.txt, .md) as a Knowledge Object.

    Creates a properly-formatted KO file with YAML frontmatter, ready for
    `kbvc add` and `kbvc commit`.

    Parameters
    ----------
    file_path  : Source file (.txt or .md).
    output_dir : Directory to write the KO into.
    ko_id      : Optional KO id; derived from filename if omitted.
    ko_type    : KO type written into frontmatter (default: "document").
    force      : Overwrite existing output file if True.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    raw_text = file_path.read_text(encoding="utf-8", errors="replace")

    derived_id = ko_id or _slugify(file_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{derived_id}.md"

    if out_path.exists() and not force:
        raise FileExistsError(
            f"Output file already exists: {out_path}\n"
            "Use --force to overwrite."
        )

    source_url = str(file_path.resolve())
    content = _make_frontmatter(derived_id, source_url, derived_id, "file") + raw_text

    out_path.write_text(content, encoding="utf-8")
    return IngestTextResult(ko_id=derived_id, output_file=out_path)
