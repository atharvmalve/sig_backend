# =============================================================================
# 1. IMPORTS
# =============================================================================

import asyncio
import json
import logging
import os
import re
import time
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
import uvicorn
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# =============================================================================
# 2. CONFIGURATION
# =============================================================================

load_dotenv()

# =============================================================================
# 3. ENVIRONMENT VARIABLES
# =============================================================================

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
APP_ENV: str = os.getenv("APP_ENV", "development")
APP_NAME: str = os.getenv("APP_NAME", "Market Urgency Engine")
APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
MAX_URLS_PER_QUERY: int = int(os.getenv("MAX_URLS_PER_QUERY", "10"))
MAX_QUERIES: int = int(os.getenv("MAX_QUERIES", "15"))

PAIN_SIGNAL_KEYWORDS: List[str] = [
    "frustrated", "frustrating", "frustration", "hate", "hating", "annoying",
    "annoyed", "struggle", "struggling", "problem", "difficult", "difficulty",
    "hard to", "impossible", "terrible", "awful", "horrible", "nightmare",
    "exhausted", "burnout", "overwhelmed", "stressed", "failing", "failed",
    "can't find", "cannot find", "no solution", "no tool", "broken", "useless",
    "waste of time", "doesn't work", "not working", "disappointing", "disappointed",
]

DEMAND_SIGNAL_KEYWORDS: List[str] = [
    "need", "looking for", "recommend", "recommendation", "alternative",
    "help", "anyone know", "does anyone", "suggest", "suggestion", "best way",
    "how to", "any app", "any tool", "any software", "want", "wish there was",
    "searching for", "tried everything", "what do you use", "seeking",
    "requirements", "must have", "request", "please help", "advice",
]

BUYING_SIGNAL_KEYWORDS: List[str] = [
    "pay", "paying", "paid", "purchase", "purchasing", "subscription",
    "worth it", "worth the money", "price", "pricing", "cost", "how much",
    "affordable", "cheap", "expensive", "budget", "buy", "buying", "license",
    "premium", "upgrade", "plan", "tier", "monthly", "annually", "fee",
]

COMPETITION_SIGNAL_KEYWORDS: List[str] = [
    "currently using", "alternative", "competitor", "existing solution",
    "compared to", "better than", "switch from", "switched from", "instead of",
    "vs", "versus", "other options", "similar to", "like X but", "unlike",
    "market leader", "dominant", "popular tool", "widely used", "standard tool",
]

TREND_SIGNAL_KEYWORDS: List[str] = [
    "growing", "popular", "increasing", "adoption", "trend", "trending",
    "rising", "more people", "everyone is", "lately", "recently", "new wave",
    "boom", "surge", "explosion", "rapidly", "fast growing", "expanding",
    "mainstream", "widely", "gaining traction", "taking off", "skyrocketing",
]

USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]

BLOCKED_DOMAINS: List[str] = [
    "google.com", "facebook.com", "twitter.com", "instagram.com",
    "tiktok.com", "youtube.com", "linkedin.com", "amazon.com",
    "ads.", "ad.", "tracker.", "analytics.",
]

MIN_CONTENT_LENGTH: int = 200

# =============================================================================
# 4. LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(APP_NAME)

# =============================================================================
# 5. MODELS
# =============================================================================

class AnalyzeRequest(BaseModel):
    problem_statement: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Description of the problem the startup solves",
        examples=["Medical students struggle with revision"],
    )
    target_user: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Who experiences this problem",
        examples=["Medical students"],
    )
    current_solution: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="How users currently solve this problem",
        examples=["Traditional notes and tutoring"],
    )
    geography: str = Field(
        default="Global",
        max_length=100,
        description="Target geography for market analysis",
        examples=["USA"],
    )

    @field_validator("problem_statement", "target_user", "current_solution", "geography")
    @classmethod
    def sanitize_input(cls, v: str) -> str:
        v = v.strip()
        v = re.sub(r"[<>\"'`;{}()\[\]\\]", "", v)
        v = re.sub(r"\s+", " ", v)
        return v


class EvidenceSummary(BaseModel):
    pain_signals: int = 0
    demand_signals: int = 0
    buying_signals: int = 0
    competition_signals: int = 0
    trend_signals: int = 0
    total_signals: int = 0
    snippets: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class Scores(BaseModel):
    pain_score: float = 0.0
    demand_score: float = 0.0
    buying_intent_score: float = 0.0
    competition_score: float = 0.0
    trend_score: float = 0.0
    market_urgency_score: float = 0.0


class Metadata(BaseModel):
    sources_scraped: int = 0
    processing_time_seconds: float = 0.0
    queries_generated: int = 0
    urls_discovered: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AnalyzeResponse(BaseModel):
    success: bool
    request: Dict[str, Any]
    generated_queries: List[str]
    evidence_summary: EvidenceSummary
    scores: Scores
    analysis: Dict[str, Any]
    metadata: Metadata


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    status: str
    service: str
    version: str

# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================

def get_rotating_user_agent(index: int = 0) -> str:
    return USER_AGENTS[index % len(USER_AGENTS)]


def is_valid_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False
        for blocked in BLOCKED_DOMAINS:
            if blocked in parsed.netloc:
                return False
        return True
    except Exception:
        return False


def clean_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    text = text.strip()
    return text


def extract_relevant_snippets(text: str, keywords: List[str], max_snippets: int = 5) -> List[str]:
    snippets: List[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    text_lower = text.lower()
    for keyword in keywords:
        if keyword.lower() in text_lower:
            for sentence in sentences:
                if keyword.lower() in sentence.lower() and len(sentence) > 30:
                    snippet = sentence.strip()[:300]
                    if snippet not in snippets:
                        snippets.append(snippet)
                    if len(snippets) >= max_snippets:
                        return snippets
    return snippets


def normalize_score(raw: float, max_val: float, weight: float = 1.0) -> float:
    if max_val <= 0:
        return 0.0
    score = min(raw / max_val, 1.0) * 100 * weight
    return round(score, 2)


def protect_prompt(text: str) -> str:
    injection_patterns = [
        r"ignore previous instructions",
        r"ignore all instructions",
        r"you are now",
        r"act as",
        r"forget everything",
        r"disregard",
        r"system prompt",
        r"jailbreak",
    ]
    text_lower = text.lower()
    for pattern in injection_patterns:
        if re.search(pattern, text_lower):
            text = re.sub(re.compile(pattern, re.IGNORECASE), "[REDACTED]", text)
    return text

# =============================================================================
# 7. SEARCH LAYER
# =============================================================================

async def search_duckduckgo(query: str, client: httpx.AsyncClient, agent_index: int = 0) -> List[str]:
    urls: List[str] = []
    try:
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        headers = {
            "User-Agent": get_rotating_user_agent(agent_index),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Referer": "https://duckduckgo.com/",
        }
        response = await client.get(search_url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            logger.warning(f"DuckDuckGo returned {response.status_code} for query: {query}")
            return urls
        soup = BeautifulSoup(response.text, "html.parser")
        results = soup.find_all("a", class_="result__url")
        if not results:
            results = soup.find_all("a", href=True)
        for tag in results:
            href = tag.get("href", "")
            if not href:
                continue
            if href.startswith("//duckduckgo.com") or "duckduckgo.com" in href:
                uddg_match = re.search(r"uddg=([^&]+)", href)
                if uddg_match:
                    href = urllib.parse.unquote(uddg_match.group(1))
            if href.startswith("http") and is_valid_url(href):
                if href not in urls:
                    urls.append(href)
            if len(urls) >= MAX_URLS_PER_QUERY:
                break
        logger.info(f"Query '{query[:50]}' → {len(urls)} URLs")
    except httpx.TimeoutException:
        logger.warning(f"Timeout searching for: {query[:50]}")
    except Exception as e:
        logger.error(f"Search error for '{query[:50]}': {e}")
    return urls


async def search_web(queries: List[str]) -> List[str]:
    all_urls: List[str] = []
    seen: set = set()
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        tasks = [
            search_duckduckgo(q, client, idx)
            for idx, q in enumerate(queries[:MAX_QUERIES])
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                for url in result:
                    if url not in seen:
                        seen.add(url)
                        all_urls.append(url)
    logger.info(f"Total unique URLs discovered: {len(all_urls)}")
    return all_urls

# =============================================================================
# 8. SCRAPING LAYER
# =============================================================================

async def fetch_page(url: str, client: httpx.AsyncClient, agent_index: int = 0) -> Optional[str]:
    try:
        headers = {
            "User-Agent": get_rotating_user_agent(agent_index),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        response = await client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code != 200:
            return None
        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return None
        return response.text
    except httpx.TimeoutException:
        logger.debug(f"Timeout fetching: {url[:80]}")
        return None
    except httpx.TooManyRedirects:
        logger.debug(f"Too many redirects: {url[:80]}")
        return None
    except Exception as e:
        logger.debug(f"Fetch error {url[:80]}: {e}")
        return None


def extract_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "meta", "link"]):
            tag.decompose()
        main_content = (
            soup.find("main") or
            soup.find("article") or
            soup.find(id=re.compile(r"content|main|post|article", re.I)) or
            soup.find(class_=re.compile(r"content|main|post|article|body", re.I)) or
            soup.find("body")
        )
        if main_content:
            text = main_content.get_text(separator=" ", strip=True)
        else:
            text = soup.get_text(separator=" ", strip=True)
        text = clean_text(text)
        return text
    except Exception as e:
        logger.debug(f"Text extraction error: {e}")
        return ""


async def scrape_sources(urls: List[str], max_pages: int = 40) -> List[Tuple[str, str]]:
    scraped: List[Tuple[str, str]] = []
    target_urls = urls[:max_pages]
    limits = httpx.Limits(max_connections=15, max_keepalive_connections=8)
    async with httpx.AsyncClient(limits=limits, follow_redirects=True) as client:
        tasks = [fetch_page(url, client, idx) for idx, url in enumerate(target_urls)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for url, result in zip(target_urls, results):
            if isinstance(result, str) and result:
                text = extract_text(result)
                if len(text) >= MIN_CONTENT_LENGTH:
                    scraped.append((url, text))
                    logger.debug(f"Scraped {len(text)} chars from {url[:60]}")
            elif isinstance(result, Exception):
                logger.debug(f"Scrape exception for {url[:60]}: {result}")
    logger.info(f"Successfully scraped {len(scraped)} pages out of {len(target_urls)} attempted")
    return scraped

# =============================================================================
# 9. EVIDENCE PROCESSING LAYER
# =============================================================================

def count_signals(text: str, keywords: List[str]) -> int:
    count = 0
    text_lower = text.lower()
    for keyword in keywords:
        occurrences = text_lower.count(keyword.lower())
        count += occurrences
    return count


def extract_behavioral_signals(scraped_pages: List[Tuple[str, str]]) -> EvidenceSummary:
    total_pain = 0
    total_demand = 0
    total_buying = 0
    total_competition = 0
    total_trend = 0
    all_snippets: List[str] = []
    sources: List[str] = []

    for url, text in scraped_pages:
        pain = count_signals(text, PAIN_SIGNAL_KEYWORDS)
        demand = count_signals(text, DEMAND_SIGNAL_KEYWORDS)
        buying = count_signals(text, BUYING_SIGNAL_KEYWORDS)
        competition = count_signals(text, COMPETITION_SIGNAL_KEYWORDS)
        trend = count_signals(text, TREND_SIGNAL_KEYWORDS)

        total_pain += pain
        total_demand += demand
        total_buying += buying
        total_competition += competition
        total_trend += trend

        if pain + demand + buying + trend > 0:
            page_snippets = extract_relevant_snippets(
                text,
                PAIN_SIGNAL_KEYWORDS + DEMAND_SIGNAL_KEYWORDS + BUYING_SIGNAL_KEYWORDS,
                max_snippets=3,
            )
            all_snippets.extend(page_snippets)
            sources.append(url)

    total_signals = total_pain + total_demand + total_buying + total_competition + total_trend

    unique_snippets = list(dict.fromkeys(all_snippets))[:20]

    logger.info(
        f"Signals extracted — Pain: {total_pain}, Demand: {total_demand}, "
        f"Buying: {total_buying}, Competition: {total_competition}, Trend: {total_trend}"
    )

    return EvidenceSummary(
        pain_signals=total_pain,
        demand_signals=total_demand,
        buying_signals=total_buying,
        competition_signals=total_competition,
        trend_signals=total_trend,
        total_signals=total_signals,
        snippets=unique_snippets,
        sources=sources[:20],
    )


def calculate_scores(evidence: EvidenceSummary, sources_count: int) -> Scores:
    page_factor = max(sources_count, 1)

    pain_raw = evidence.pain_signals / page_factor
    demand_raw = evidence.demand_signals / page_factor
    buying_raw = evidence.buying_signals / page_factor
    competition_raw = evidence.competition_signals / page_factor
    trend_raw = evidence.trend_signals / page_factor

    PAIN_MAX = 15.0
    DEMAND_MAX = 20.0
    BUYING_MAX = 5.0
    COMPETITION_MAX = 10.0
    TREND_MAX = 8.0

    pain_score = normalize_score(pain_raw, PAIN_MAX, weight=1.0)
    demand_score = normalize_score(demand_raw, DEMAND_MAX, weight=1.0)
    buying_intent_score = normalize_score(buying_raw, BUYING_MAX, weight=1.0)
    competition_score = normalize_score(competition_raw, COMPETITION_MAX, weight=1.0)
    trend_score = normalize_score(trend_raw, TREND_MAX, weight=1.0)

    # Market urgency = weighted composite
    # Pain (30%) + Demand (30%) + Buying Intent (20%) + Trend (15%) + Competition modifier (5%)
    competition_modifier = (100 - competition_score) * 0.05
    market_urgency_score = round(
        (pain_score * 0.30) +
        (demand_score * 0.30) +
        (buying_intent_score * 0.20) +
        (trend_score * 0.15) +
        competition_modifier,
        2,
    )
    market_urgency_score = min(market_urgency_score, 100.0)

    return Scores(
        pain_score=pain_score,
        demand_score=demand_score,
        buying_intent_score=buying_intent_score,
        competition_score=competition_score,
        trend_score=trend_score,
        market_urgency_score=market_urgency_score,
    )

# =============================================================================
# 10. GEMINI INTEGRATION
# =============================================================================

def configure_gemini() -> Optional[Any]:
    if not GEMINI_AVAILABLE:
        logger.warning("google-generativeai not installed. Gemini unavailable.")
        return None
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set. Gemini unavailable.")
        return None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        logger.info(f"Gemini configured with model: {GEMINI_MODEL}")
        return model
    except Exception as e:
        logger.error(f"Failed to configure Gemini: {e}")
        return None


gemini_model: Optional[Any] = None


async def generate_queries(
    problem_statement: str,
    target_user: str,
    current_solution: str,
    geography: str,
) -> List[str]:
    problem_statement = protect_prompt(problem_statement)
    target_user = protect_prompt(target_user)
    current_solution = protect_prompt(current_solution)
    geography = protect_prompt(geography)

    if gemini_model is None:
        logger.warning("Gemini unavailable. Using fallback query generation.")
        return _fallback_queries(problem_statement, target_user, geography)

    prompt = f"""You are a market research expert. Generate exactly 12 diverse, highly targeted web search queries to validate market demand for a startup idea.

Problem: {problem_statement}
Target Users: {target_user}
Current Solution: {current_solution}
Geography: {geography}

Generate queries that will find:
1. User frustrations and pain points about this problem
2. People actively asking for help or solutions
3. Reddit, Quora, or forum discussions about this problem
4. People expressing buying intent or willingness to pay
5. Existing tools or competitors users mention
6. Growing trends in this market

Return ONLY a JSON array of strings. No explanations, no markdown, just the JSON array.
Example format: ["query 1", "query 2", "query 3"]

Each query should be 3-8 words, natural-language, as a real user would search."""

    try:
        logger.info("Calling Gemini for query generation")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(prompt),
        )
        raw = response.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        queries = json.loads(raw)
        if isinstance(queries, list):
            queries = [str(q).strip() for q in queries if isinstance(q, str) and q.strip()]
            logger.info(f"Gemini generated {len(queries)} queries")
            return queries[:MAX_QUERIES]
    except json.JSONDecodeError as e:
        logger.warning(f"Gemini query JSON parse error: {e}. Using fallback.")
    except Exception as e:
        logger.error(f"Gemini query generation error: {e}. Using fallback.")

    return _fallback_queries(problem_statement, target_user, geography)


def _fallback_queries(problem_statement: str, target_user: str, geography: str) -> List[str]:
    problem_short = problem_statement[:60]
    user_short = target_user[:40]
    geo = geography if geography.lower() != "global" else ""
    geo_suffix = f" {geo}" if geo else ""

    queries = [
        f"{user_short} frustration{geo_suffix}",
        f"{user_short} struggling with{geo_suffix}",
        f"{user_short} problems{geo_suffix}",
        f"help for {user_short}{geo_suffix}",
        f"best tools for {user_short}{geo_suffix}",
        f"{user_short} alternatives{geo_suffix}",
        f"reddit {user_short} advice{geo_suffix}",
        f"{user_short} recommendation{geo_suffix}",
        f"{problem_short[:40]} solution",
        f"{user_short} forum discussion",
        f"{user_short} tips{geo_suffix}",
        f"how {user_short} solve{geo_suffix}",
    ]
    logger.info(f"Fallback generated {len(queries)} queries")
    return queries


async def generate_market_analysis(
    request: AnalyzeRequest,
    evidence: EvidenceSummary,
    scores: Scores,
) -> Dict[str, Any]:
    default_analysis = {
        "market_verdict": "Insufficient data for analysis",
        "confidence": 0,
        "top_pain_points": [],
        "top_opportunities": [],
        "key_competitors": [],
        "recommended_next_steps": [],
    }

    if gemini_model is None:
        logger.warning("Gemini unavailable. Returning rule-based analysis.")
        return _rule_based_analysis(evidence, scores)

    snippets_sample = "\n".join(f"- {s}" for s in evidence.snippets[:10])
    if not snippets_sample:
        snippets_sample = "No direct user quotes scraped."

    prompt = f"""You are an elite startup analyst, venture capitalist, and market researcher.

STARTUP CONTEXT:
- Problem: {protect_prompt(request.problem_statement)}
- Target Users: {protect_prompt(request.target_user)}
- Current Solution: {protect_prompt(request.current_solution)}
- Geography: {protect_prompt(request.geography)}

BEHAVIORAL SIGNAL COUNTS (from real internet scraping):
- Pain signals found: {evidence.pain_signals}
- Demand signals found: {evidence.demand_signals}
- Buying intent signals found: {evidence.buying_signals}
- Competition signals found: {evidence.competition_signals}
- Trend signals found: {evidence.trend_signals}
- Total pages with signals: {len(evidence.sources)}

MARKET SCORES (0-100):
- Pain Score: {scores.pain_score}
- Demand Score: {scores.demand_score}
- Buying Intent Score: {scores.buying_intent_score}
- Competition Score: {scores.competition_score}
- Trend Score: {scores.trend_score}
- Market Urgency Score: {scores.market_urgency_score}

REAL INTERNET SNIPPETS (from scraped sources):
{snippets_sample}

Analyze this startup idea strictly based on the evidence above. Do not invent or assume any information not present in the signals.

Return ONLY a valid JSON object with this exact schema:
{{
  "market_verdict": "string (e.g., Strong Demand, Moderate Demand, Low Demand, Insufficient Evidence)",
  "confidence": integer (0-100 based on signal volume and score strength),
  "top_pain_points": ["string", "string", "string"],
  "top_opportunities": ["string", "string", "string"],
  "key_competitors": ["string", "string"],
  "recommended_next_steps": ["string", "string", "string"]
}}

No markdown. No explanations. Just the JSON object."""

    try:
        logger.info("Calling Gemini for market analysis")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_model.generate_content(prompt),
        )
        raw = response.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        analysis = json.loads(raw)
        if isinstance(analysis, dict):
            for key in default_analysis:
                if key not in analysis:
                    analysis[key] = default_analysis[key]
            logger.info("Gemini market analysis completed")
            return analysis
    except json.JSONDecodeError as e:
        logger.warning(f"Gemini analysis JSON parse error: {e}. Using rule-based fallback.")
    except Exception as e:
        logger.error(f"Gemini analysis error: {e}. Using rule-based fallback.")

    return _rule_based_analysis(evidence, scores)


def _rule_based_analysis(evidence: EvidenceSummary, scores: Scores) -> Dict[str, Any]:
    urgency = scores.market_urgency_score

    if urgency >= 70:
        verdict = "Strong Demand"
        confidence = min(int(urgency), 90)
    elif urgency >= 45:
        verdict = "Moderate Demand"
        confidence = int(urgency * 0.8)
    elif urgency >= 20:
        verdict = "Low Demand"
        confidence = int(urgency * 0.7)
    else:
        verdict = "Insufficient Evidence"
        confidence = int(urgency * 0.5)

    pain_points = []
    if evidence.pain_signals > 5:
        pain_points.append(f"Significant user frustration detected ({evidence.pain_signals} pain signals)")
    if evidence.demand_signals > 10:
        pain_points.append(f"High unmet need ({evidence.demand_signals} demand signals)")
    if evidence.buying_signals > 3:
        pain_points.append(f"Willingness to pay signals detected ({evidence.buying_signals} buying signals)")

    opportunities = []
    if scores.demand_score > 50:
        opportunities.append("High user demand creates strong acquisition opportunity")
    if scores.competition_score < 40:
        opportunities.append("Low competition score suggests market gap")
    if scores.trend_score > 40:
        opportunities.append("Growing trend signals indicate expanding market")
    if not opportunities:
        opportunities.append("Validate with direct user interviews before investing")

    next_steps = [
        "Conduct 20+ user interviews to validate pain points",
        "Build a landing page to test conversion and buying intent",
        "Identify 3 direct competitors and map their weaknesses",
    ]

    return {
        "market_verdict": verdict,
        "confidence": confidence,
        "top_pain_points": pain_points or ["Limited pain signal data — run more targeted searches"],
        "top_opportunities": opportunities,
        "key_competitors": ["Research required — insufficient competition data scraped"],
        "recommended_next_steps": next_steps,
    }

# =============================================================================
# 11. BUSINESS LOGIC
# =============================================================================

async def run_market_validation(request: AnalyzeRequest) -> AnalyzeResponse:
    start_time = time.time()

    logger.info(
        f"Starting market validation | Problem: '{request.problem_statement[:60]}' | "
        f"Target: '{request.target_user}' | Geo: '{request.geography}'"
    )

    # Step 1: Generate search queries
    queries = await generate_queries(
        request.problem_statement,
        request.target_user,
        request.current_solution,
        request.geography,
    )
    logger.info(f"Generated {len(queries)} search queries")

    # Step 2: Search the web
    all_urls = await search_web(queries)
    logger.info(f"Discovered {len(all_urls)} unique URLs")

    # Step 3: Scrape sources
    scraped_pages = await scrape_sources(all_urls)
    logger.info(f"Scraped {len(scraped_pages)} pages successfully")

    # Step 4: Extract behavioral signals
    evidence = extract_behavioral_signals(scraped_pages)

    # Step 5: Calculate scores
    scores = calculate_scores(evidence, len(scraped_pages))
    logger.info(f"Market Urgency Score: {scores.market_urgency_score}")

    # Step 6: Generate AI analysis
    analysis = await generate_market_analysis(request, evidence, scores)

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Market validation completed in {elapsed}s")

    return AnalyzeResponse(
        success=True,
        request=request.model_dump(),
        generated_queries=queries,
        evidence_summary=evidence,
        scores=scores,
        analysis=analysis,
        metadata=Metadata(
            sources_scraped=len(scraped_pages),
            processing_time_seconds=elapsed,
            queries_generated=len(queries),
            urls_discovered=len(all_urls),
        ),
    )

# =============================================================================
# 12. API ROUTES
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gemini_model
    logger.info(f"Starting {APP_NAME} v{APP_VERSION} [{APP_ENV}]")
    gemini_model = configure_gemini()
    yield
    logger.info(f"Shutting down {APP_NAME}")


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "Market Urgency Engine — Validate startup ideas using real-world behavioral signals "
        "collected from public internet discussions. Discover evidence of demand, pain points, "
        "buying intent, competition, and market trends."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if APP_ENV == "development" else "An unexpected error occurred.",
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning(f"HTTP {exc.status_code} on {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            detail=exc.detail,
        ).model_dump(),
    )


@app.get(
    "/",
    response_model=RootResponse,
    summary="Root health check",
    tags=["Health"],
)
async def root() -> RootResponse:
    return RootResponse(
        status="healthy",
        service=APP_NAME,
        version=APP_VERSION,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Health"],
)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze market demand for a startup idea",
    description=(
        "Accepts a startup problem statement and target user context. "
        "Searches the web for real-world discussions, scrapes behavioral signals, "
        "scores market demand, and returns an evidence-backed validation report."
    ),
    tags=["Analysis"],
    responses={
        200: {"description": "Successful market analysis"},
        422: {"description": "Validation error — invalid request body"},
        500: {"description": "Internal server error"},
    },
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    logger.info(f"POST /analyze — problem='{request.problem_statement[:60]}'")
    try:
        result = await run_market_validation(request)
        return result
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.error("Analysis timed out")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Analysis timed out. Try a more specific problem statement.",
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )

# =============================================================================
# 13. STARTUP EVENT
# =============================================================================

# Gemini initialization is handled in the lifespan context manager above.
# This section is reserved for any additional startup validation.

def _validate_environment() -> None:
    if not GEMINI_API_KEY:
        logger.warning(
            "GEMINI_API_KEY is not set. Gemini features will be disabled. "
            "Rule-based fallbacks will be used for query generation and analysis."
        )
    logger.info(f"Environment: {APP_ENV}")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT}s")
    logger.info(f"Max URLs per query: {MAX_URLS_PER_QUERY}")
    logger.info(f"Max queries: {MAX_QUERIES}")

# =============================================================================
# 14. MAIN ENTRY
# =============================================================================

if __name__ == "__main__":
    _validate_environment()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=APP_ENV == "development",
        log_level="info",
        access_log=True,
    )