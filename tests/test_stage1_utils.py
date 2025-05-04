#!/usr/bin/env python3
import pytest

# Need to make the script importable. For now, assume it's possible.
# This might require adding scripts/ to sys.path or creating a package structure.
# For simplicity, let's assume direct import works for now.
from scripts.stage1_scan_warc import extract_domain

# --- Tests for extract_domain ---

@pytest.mark.parametrize(
    "url, expected_domain, expected_alt_domain",
    [
        ("http://www.example.com/path", "www.example.com", "example.com"),
        ("https://example.com/path?query=1", "example.com", "www.example.com"),
        ("http://example.co.uk/", "example.co.uk", "www.example.co.uk"),
        ("http://subdomain.example.com", "subdomain.example.com", "www.subdomain.example.com"),
        ("https://www.sub.example.co.uk/a/b", "www.sub.example.co.uk", "sub.example.co.uk"),
        ("http://192.168.1.1/page", "192.168.1.1", "www.192.168.1.1"), # IP addresses
        ("ftp://ftp.example.com/", "ftp.example.com", "www.ftp.example.com"),
        ("http://example.com:8080/", "example.com", "www.example.com"), # With port
        ("http://WwW.ExAmPlE.cOm/", "www.example.com", "example.com"), # Case insensitivity
        ("dns:example.com", "example.com", "www.example.com"), # URI scheme without //
        ("//example.com/path", "example.com", "www.example.com"), # Protocol-relative URL
        # Edge cases & Invalid inputs
        ("justadomain", None, None), # No TLD-like structure
        ("http://localhost/", "localhost", "www.localhost"), # Localhost
        ("http:///path", None, None), # Empty domain
        ("http://", None, None), # No domain
        ("", None, None), # Empty string
        (None, None, None), # None input
        ("http://example.com.", "example.com.", "www.example.com."), # Trailing dot
        ("file:///path/to/file/example.com", "example.com", "www.example.com"), # File URI (might depend on implementation)
    ]
)
def test_extract_domain(url, expected_domain, expected_alt_domain):
    domain, alt_domain = extract_domain(url)
    assert domain == expected_domain
    assert alt_domain == expected_alt_domain 