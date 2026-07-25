import pytest
from pathlib import Path
from src.ingestion.loader import load_file, load_directory, UnsupportedFormatError


TEST_MD = Path("data/raw/cross_border_registration.md")
TEST_DIR = Path("data/raw")


class TestLoadFile:
    def test_load_md_returns_text(self):
        text, meta = load_file(str(TEST_MD))
        assert len(text) > 100
        assert "跨境投资登记系统" in text

    def test_load_md_returns_metadata(self):
        text, meta = load_file(str(TEST_MD))
        assert meta["doc_id"] == "cross_border_registration"
        assert meta["source_type"] == "MD"
        assert meta["file_size"] > 0

    def test_load_non_existent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_file("data/raw/nonexistent.md")

    def test_load_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("dummy")
        with pytest.raises(UnsupportedFormatError):
            load_file(str(f))


class TestLoadDirectory:
    def test_load_directory_returns_multiple(self):
        results = load_directory(str(TEST_DIR))
        assert len(results) >= 10

    def test_each_result_is_text_meta_tuple(self):
        results = load_directory(str(TEST_DIR))
        for text, meta in results:
            assert isinstance(text, str)
            assert isinstance(meta, dict)
            assert "doc_id" in meta

    def test_load_invalid_directory_raises(self):
        with pytest.raises(NotADirectoryError):
            load_directory("data/raw/nonexistent")


class TestDecodeEdgeCases:
    def test_load_md_with_special_chars(self):
        """验证带特殊字符的 .md 能正常解码"""
        text, meta = load_file("data/raw/foreign_exchange_management.md")
        assert len(text) > 50

    def test_load_empty_file_returns_empty(self):
        """验证空文件处理"""
        path = Path("data/raw/__empty_test.md")
        try:
            path.write_text("", encoding="utf-8")
            text, meta = load_file(str(path))
            assert text == ""
        finally:
            if path.exists():
                path.unlink()
