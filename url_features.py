"""
Extract UCI Phishing Websites–style features from a raw URL.

Feature values follow the dataset convention:
  1 = Legitimate, 0 = Suspicious, -1 = Phishing.

Address-bar features are always computed from the URL string.
Optional page fetch fills HTML/abnormal features when possible.
External signals (Alexa, PageRank, Google Index, WHOIS age) that are
unavailable today are set to a neutral/suspicious default (0) and listed
in the returned `approximated` set.
"""
from __future__ import annotations

import ipaddress
import re
import socket
import ssl
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

# UCI / Mohammad et al. encoding
LEGIT, SUSPICIOUS, PHISH = 1, 0, -1

FEATURE_ORDER = [
    "having_ip_address",
    "url_length",
    "shortining_service",
    "having_at_symbol",
    "double_slash_redirecting",
    "prefix_suffix",
    "having_sub_domain",
    "sslfinal_state",
    "domain_registration_length",
    "favicon",
    "port",
    "https_token",
    "request_url",
    "url_of_anchor",
    "links_in_tags",
    "sfh",
    "submitting_to_email",
    "abnormal_url",
    "redirect",
    "on_mouseover",
    "rightclick",
    "popupwindow",
    "iframe",
    "age_of_domain",
    "dnsrecord",
    "web_traffic",
    "page_rank",
    "google_index",
    "links_pointing_to_page",
    "statistical_report",
]

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "mcaf.ee", "rebrand.ly", "cutt.ly", "shorturl.at",
    "tiny.cc", "rb.gy", "s.id", "v.gd", "tr.im", "cli.gs", "short.link",
}

# Prefer status for common ports (paper Table 1): open only 80/443.
PREFERRED_OPEN = {80, 443}
PREFERRED_CLOSED = {21, 22, 23, 445, 1433, 1521, 3306, 3389}

TRUSTED_ISSUER_HINTS = (
    "geotrust", "godaddy", "network solutions", "thawte", "comodo",
    "doster", "verisign", "digicert", "let's encrypt", "lets encrypt",
    "sectigo", "globalsign", "amazon", "google trust", "microsoft",
    "entrust", "usertrust", "ssl.com", "buypass",
)

EXTERNAL_DEFAULTS = {
    "domain_registration_length",
    "age_of_domain",
    "web_traffic",
    "page_rank",
    "google_index",
    "links_pointing_to_page",
    "statistical_report",
}

FETCH_TIMEOUT = 8
MAX_HTML_BYTES = 500_000
USER_AGENT = "PhishingDetectorAcademic/1.0 (+educational; not a crawler)"


class _PageScan(HTMLParser):
    """Collect tags/attrs needed for HTML-based UCI features."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.anchors: list[str] = []
        self.requests: list[str] = []  # img/script/link/source/iframe src/href
        self.meta_script_link: list[str] = []
        self.forms: list[str] = []
        self.has_mailto = False
        self.has_mail_fn = False
        self.favicon_hrefs: list[str] = []
        self.on_mouseover_status = False
        self.right_click_disabled = False
        self.popup_with_input = False
        self.has_iframe = False
        self._in_script = False
        self._script_buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        ad = {k.lower(): (v or "") for k, v in attrs if k}
        tag = tag.lower()

        if tag == "a":
            self.anchors.append(ad.get("href", ""))
            if "onmouseover" in ad and "window.status" in ad["onmouseover"].lower():
                self.on_mouseover_status = True
        elif tag in {"img", "audio", "video", "embed", "source", "iframe"}:
            src = ad.get("src") or ad.get("data-src") or ""
            if src:
                self.requests.append(src)
            if tag == "iframe":
                self.has_iframe = True
        elif tag == "script":
            self._in_script = True
            src = ad.get("src", "")
            if src:
                self.requests.append(src)
                self.meta_script_link.append(src)
        elif tag == "link":
            href = ad.get("href", "")
            if href:
                self.meta_script_link.append(href)
                rel = ad.get("rel", "").lower()
                if "icon" in rel:
                    self.favicon_hrefs.append(href)
                else:
                    self.requests.append(href)
        elif tag == "meta":
            content = ad.get("content", "")
            if content.startswith("http"):
                self.meta_script_link.append(content)
        elif tag == "form":
            self.forms.append(ad.get("action", ""))
            if ad.get("action", "").lower().startswith("mailto:"):
                self.has_mailto = True
        elif tag == "input" and ad.get("type", "").lower() in {"password", "text", "email"}:
            # Heuristic: inputs inside popups are hard to detect; flag suspicious JS below.
            pass

        for key in ("onclick", "onmouseover", "onmousedown"):
            if key in ad:
                low = ad[key].lower()
                if "window.status" in low:
                    self.on_mouseover_status = True
                if "event.button" in low and "2" in low:
                    self.right_click_disabled = True
                if "window.open" in low and ("input" in low or "prompt" in low):
                    self.popup_with_input = True

    def handle_endtag(self, tag):
        if tag.lower() == "script" and self._in_script:
            self._in_script = False
            text = "".join(self._script_buf).lower()
            self._script_buf.clear()
            if "mail(" in text:
                self.has_mail_fn = True
            if "window.status" in text:
                self.on_mouseover_status = True
            if "event.button" in text and "==2" in text.replace(" ", ""):
                self.right_click_disabled = True
            if "window.open" in text and ("password" in text or "<input" in text):
                self.popup_with_input = True

    def handle_data(self, data):
        if self._in_script:
            self._script_buf.append(data)


def _normalize_url(raw: str) -> str:
    url = raw.strip()
    if not url:
        raise ValueError("Please enter a URL.")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) is None:
        url = "http://" + url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are supported.")
    if not parsed.netloc:
        raise ValueError("Could not parse a host from that URL.")
    return url


def _host(parsed) -> str:
    host = parsed.hostname or ""
    return host.lower().strip(".")


def _domain_has_ip(host: str) -> bool:
    if not host:
        return False
    # Strip brackets for IPv6 literals.
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        pass
    # Hex-encoded IP style used in the paper.
    if re.fullmatch(r"(0x[0-9a-f]+(?:\.0x[0-9a-f]+){3})", candidate, re.I):
        return True
    return False


def _ratio_bucket(pct: float, low: float, high: float) -> int:
    if pct < low:
        return LEGIT
    if pct <= high:
        return SUSPICIOUS
    return PHISH


def _same_domain(base_host: str, link: str, page_url: str) -> bool | None:
    """True if link stays on base host; None if empty / non-navigating."""
    link = (link or "").strip()
    if not link or link.startswith(("#", "javascript:", "mailto:", "tel:", "data:")):
        return None
    absolute = urljoin(page_url, link)
    other = urlparse(absolute).hostname
    if not other:
        return None
    other = other.lower()
    base = base_host.lower()
    return other == base or other.endswith("." + base)


def _address_bar_features(url: str, parsed) -> dict[str, int]:
    host = _host(parsed)
    path_url = url

    having_ip = PHISH if _domain_has_ip(host) else LEGIT

    length = len(url)
    if length < 54:
        url_length = LEGIT
    elif length <= 75:
        url_length = SUSPICIOUS
    else:
        url_length = PHISH

    shortining = PHISH if host in SHORTENERS or any(
        host.endswith("." + s) for s in SHORTENERS
    ) else LEGIT

    having_at = PHISH if "@" in path_url else LEGIT

    last_slash = path_url.rfind("//")
    double_slash = PHISH if last_slash > 7 else LEGIT

    prefix_suffix = PHISH if "-" in host else LEGIT

    # Dots in domain after dropping leading www.
    domain = host[4:] if host.startswith("www.") else host
    dots = domain.count(".")
    if dots <= 1:
        having_sub = LEGIT
    elif dots == 2:
        having_sub = SUSPICIOUS
    else:
        having_sub = PHISH

    port_num = parsed.port
    if port_num is None:
        port_feat = LEGIT
    elif port_num in PREFERRED_OPEN:
        port_feat = LEGIT
    elif port_num in PREFERRED_CLOSED:
        port_feat = PHISH
    else:
        port_feat = PHISH

    # Paper: "HTTPS" token in the domain part (e.g. https-www-paypal...).
    https_token = PHISH if re.search(r"https?", host, re.I) else LEGIT

    return {
        "having_ip_address": having_ip,
        "url_length": url_length,
        "shortining_service": shortining,
        "having_at_symbol": having_at,
        "double_slash_redirecting": double_slash,
        "prefix_suffix": prefix_suffix,
        "having_sub_domain": having_sub,
        "port": port_feat,
        "https_token": https_token,
    }


def _ssl_feature(host: str, use_https: bool) -> int:
    if not use_https or not host or _domain_has_ip(host):
        return PHISH
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
        if not cert:
            return SUSPICIOUS
        issuer = " ".join(
            v for tup in cert.get("issuer", ()) for _, v in tup
        ).lower()
        trusted = any(h in issuer for h in TRUSTED_ISSUER_HINTS)
        # Original paper required certificate age ≥ 1 year. Modern CAs (esp.
        # Let's Encrypt) issue short-lived certs, so we only require a currently
        # valid certificate from a known issuer.
        not_after = cert.get("notAfter")
        valid_now = True
        if not_after:
            try:
                end = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(
                    tzinfo=timezone.utc
                )
                valid_now = end > datetime.now(timezone.utc)
            except ValueError:
                valid_now = True
        if trusted and valid_now:
            return LEGIT
        if use_https and not trusted:
            return SUSPICIOUS
        return PHISH
    except OSError:
        return PHISH


def _dns_feature(host: str) -> int:
    if not host or _domain_has_ip(host):
        return LEGIT  # IP hosts skip DNS name lookup semantics
    try:
        socket.getaddrinfo(host, None)
        return LEGIT
    except socket.gaierror:
        return PHISH


def _html_features(page_url: str, host: str, html: str, redirect_count: int) -> dict[str, int]:
    scanner = _PageScan()
    try:
        scanner.feed(html)
        scanner.close()
    except Exception:
        pass

    def external_ratio(links: list[str]) -> float | None:
        scored = [_same_domain(host, link, page_url) for link in links]
        usable = [s for s in scored if s is not None]
        if not usable:
            return None
        external = sum(1 for s in usable if s is False)
        return 100.0 * external / len(usable)

    req_pct = external_ratio(scanner.requests)
    request_url = _ratio_bucket(req_pct, 22, 61) if req_pct is not None else LEGIT

    # Anchors: empty / # / javascript count as phishing-like anchors in the paper.
    anchor_scores = []
    for a in scanner.anchors:
        a = (a or "").strip()
        if not a or a.startswith(("#", "javascript:", "mailto:")):
            anchor_scores.append(True)  # treat as "bad" / external-like
        else:
            same = _same_domain(host, a, page_url)
            anchor_scores.append(same is False)
    if anchor_scores:
        bad_pct = 100.0 * sum(anchor_scores) / len(anchor_scores)
        url_of_anchor = _ratio_bucket(bad_pct, 31, 67)
    else:
        url_of_anchor = LEGIT

    tag_pct = external_ratio(scanner.meta_script_link)
    links_in_tags = _ratio_bucket(tag_pct, 17, 81) if tag_pct is not None else LEGIT

    if not scanner.forms:
        sfh = LEGIT
    else:
        flags = []
        for action in scanner.forms:
            action = (action or "").strip()
            if action == "" or action.lower() == "about:blank":
                flags.append(PHISH)
            else:
                same = _same_domain(host, action, page_url)
                if same is False:
                    flags.append(SUSPICIOUS)
                else:
                    flags.append(LEGIT)
        sfh = min(flags)

    submitting = PHISH if (scanner.has_mailto or scanner.has_mail_fn) else LEGIT

    # Favicon from external domain → phishing.
    favicon = LEGIT
    for href in scanner.favicon_hrefs:
        same = _same_domain(host, href, page_url)
        if same is False:
            favicon = PHISH
            break

    if redirect_count <= 1:
        redirect = LEGIT
    elif redirect_count < 4:
        redirect = SUSPICIOUS
    else:
        redirect = PHISH

    return {
        "favicon": favicon,
        "request_url": request_url,
        "url_of_anchor": url_of_anchor,
        "links_in_tags": links_in_tags,
        "sfh": sfh,
        "submitting_to_email": submitting,
        "redirect": redirect,
        "on_mouseover": PHISH if scanner.on_mouseover_status else LEGIT,
        "rightclick": PHISH if scanner.right_click_disabled else LEGIT,
        "popupwindow": PHISH if scanner.popup_with_input else LEGIT,
        "iframe": PHISH if scanner.has_iframe else LEGIT,
        # Without WHOIS identity matching, assume hostname present in URL is normal.
        "abnormal_url": LEGIT,
    }


def _fetch_page(url: str) -> tuple[str | None, str | None, int, str | None]:
    """Returns (final_url, html, redirect_count, error)."""
    try:
        session = requests.Session()
        session.max_redirects = 5
        resp = session.get(
            url,
            timeout=FETCH_TIMEOUT,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            allow_redirects=True,
            stream=True,
        )
        redirect_count = len(resp.history)
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "html" not in content_type and "text/" not in content_type and content_type:
            return resp.url, None, redirect_count, f"Non-HTML response ({content_type})"
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=16384):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_HTML_BYTES:
                break
            chunks.append(chunk)
        html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="replace")
        return resp.url, html, redirect_count, None
    except requests.TooManyRedirects:
        return None, None, 5, "Too many redirects"
    except requests.RequestException as exc:
        return None, None, 0, str(exc)


def extract_features(raw_url: str, fetch_page: bool = True) -> dict[str, Any]:
    """
    Extract the 30 UCI features for ``raw_url``.

    Returns dict with keys:
      features: dict[str, int]
      approximated: sorted list of features that used defaults / heuristics
      url: normalized URL
      fetch_error: optional error string from page fetch
      fetched: whether HTML was obtained
    """
    url = _normalize_url(raw_url)
    parsed = urlparse(url)
    host = _host(parsed)
    features: dict[str, int] = {}
    approximated: set[str] = set()

    features.update(_address_bar_features(url, parsed))
    features["sslfinal_state"] = _ssl_feature(host, parsed.scheme == "https")
    features["dnsrecord"] = _dns_feature(host)

    # Defaults for HTML / domain features before optional enrichments.
    html_defaults = {
        "favicon": LEGIT,
        "request_url": SUSPICIOUS,
        "url_of_anchor": SUSPICIOUS,
        "links_in_tags": SUSPICIOUS,
        "sfh": SUSPICIOUS,
        "submitting_to_email": LEGIT,
        "abnormal_url": LEGIT,
        "redirect": LEGIT,
        "on_mouseover": LEGIT,
        "rightclick": LEGIT,
        "popupwindow": LEGIT,
        "iframe": LEGIT,
    }
    features.update(html_defaults)

    for name in EXTERNAL_DEFAULTS:
        features[name] = SUSPICIOUS
        approximated.add(name)

    fetch_error = None
    fetched = False
    if fetch_page:
        final_url, html, redirect_count, fetch_error = _fetch_page(url)
        if html and final_url:
            fetched = True
            page_host = _host(urlparse(final_url)) or host
            features.update(_html_features(final_url, page_host, html, redirect_count))
            # These were filled from HTML — no longer pure defaults (except abnormal_url).
            for name in (
                "favicon", "request_url", "url_of_anchor", "links_in_tags", "sfh",
                "submitting_to_email", "redirect", "on_mouseover", "rightclick",
                "popupwindow", "iframe",
            ):
                approximated.discard(name)
            approximated.add("abnormal_url")
        else:
            approximated.update(html_defaults)
            if redirect_count:
                if redirect_count <= 1:
                    features["redirect"] = LEGIT
                elif redirect_count < 4:
                    features["redirect"] = SUSPICIOUS
                else:
                    features["redirect"] = PHISH
    else:
        approximated.update(html_defaults)

    # Ensure full ordered feature set.
    ordered = {name: int(features[name]) for name in FEATURE_ORDER}
    return {
        "features": ordered,
        "approximated": sorted(approximated),
        "url": url,
        "fetch_error": fetch_error,
        "fetched": fetched,
    }
