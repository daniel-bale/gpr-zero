#!/usr/bin/env python3
import pytest

# Need to make the script importable. For now, assume it's possible.
# This might require adding scripts/ to sys.path or creating a package structure.
# For simplicity, let's assume direct import works for now.
from scripts.stage1_scan_warc import extract_domain

# --- Tests for extract_domain ---

@pytest.mark.parametrize(
    "url, expected_domain",
    [
        ("http://www.example.com/path", "example.com"),
        ("https://example.com/path?query=1", "example.com"),
        ("http://example.co.uk/", "example.co.uk"),
        ("http://subdomain.example.com", "subdomain.example.com"),
        ("https://www.sub.example.co.uk/a/b", "sub.example.co.uk"),
        ("http://192.168.1.1/page", "192.168.1.1"),
        ("ftp://ftp.example.com/", "ftp.example.com"),
        ("http://example.com:8080/", "example.com"),
        ("http://WwW.ExAmPlE.cOm/", "example.com"),
        ("dns:example.com", "example.com"),
        ("//example.com/path", "example.com"),
        # Edge cases & Invalid inputs
        ("justadomain", None),
        ("http://localhost/", "localhost"),
        ("http:///path", None),
        ("http://", None),
        ("", None),
        (None, None),
        ("http://example.com.", "example.com."),
        ("file:///path/to/file/example.com", "example.com"),
        ("http://www.", None),
    ]
)
def test_extract_domain(url, expected_domain):
    domain = extract_domain(url)
    assert domain == expected_domain 