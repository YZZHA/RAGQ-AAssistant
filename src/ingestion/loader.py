# input:  file path, file format (MD / TXT / PDF / DOCX)
# output: (text: str, metadata: dict)
# pos:    摄入管道 → 文档加载器，将源文件转为可处理文本

from pathlib import Path
from typing import Tuple


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}


class UnsupportedFormatError(ValueError):
    pass


class FileNotFoundError_(FileNotFoundError):
    pass


def load_file(file_path: str) -> Tuple[str, dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError_(f"文件不存在: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"不支持的格式: {ext}，支持: {SUPPORTED_EXTENSIONS}")

    raw = path.read_bytes()
    text = _decode_text(raw, ext)

    metadata = {
        "doc_id": path.stem,
        "filename": path.name,
        "source_type": ext.lstrip(".").upper(),
        "file_size": len(raw),
    }
    return text, metadata


def load_directory(dir_path: str) -> list[Tuple[str, dict]]:
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"目录不存在: {dir_path}")

    results = []
    for f in sorted(path.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                text, meta = load_file(str(f))
                results.append((text, meta))
            except Exception as e:
                print(f"  [跳过] {f.name}: {e}")
    return results


def _decode_text(raw: bytes, ext: str) -> str:
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(raw)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise ImportError("需要安装 pypdf: pip install pypdf")

    if ext == ".docx":
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            raise ImportError("需要安装 python-docx: pip install python-docx")

    # .md / .txt — 直接 UTF-8 解码
    return raw.decode("utf-8")
