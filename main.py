from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import re
from typing import List, Optional, Dict, Any

app = FastAPI(title="LinkedIn Papers Resolver")

# CORS: allow your extension + localhost.
# We'll also allow "*" to keep setup easy. You can tighten later.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENALEX = "https://api.openalex.org"
S2 = "https://api.semanticscholar.org/graph/v1"


class ResolveRequest(BaseModel):
    linkedin_url: Optional[str] = None
    name: str = Field(..., min_length=2)
    headline: Optional[str] = ""
    # Optional extra signals (extension can send later)
    experience: Optional[List[Dict[str, Any]]] = None
    education: Optional[List[Dict[str, Any]]] = None
    links: Optional[List[str]] = None


def tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(t) >= 4]


def overlap(a: List[str], b: List[str]) -> int:
    sb = set(b)
    return sum(1 for t in a if t in sb)


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def label(score: float) -> str:
    if score >= 0.85:
        return "High"
    if score >= 0.70:
        return "Medium"
    if score >= 0.55:
        return "Low"
    return "Very Low"


async def openalex_author_search(client: httpx.AsyncClient, name: str) -> List[dict]:
    r = await client.get(f"{OPENALEX}/authors", params={"search": name, "per-page": 5})
    r.raise_for_status()
    return r.json().get("results", [])


async def openalex_top_works(client: httpx.AsyncClient, author_id: str) -> List[dict]:
    # author_id is a URL like https://openalex.org/A123...
    r = await client.get(
        f"{OPENALEX}/works",
        params={
            "filter": f"authorships.author.id:{author_id}",
            "per-page": 3,
            "sort": "cited_by_count:desc",
        },
    )
    r.raise_for_status()
    return r.json().get("results", [])


async def s2_author_search(client: httpx.AsyncClient, name: str) -> List[dict]:
    r = await client.get(
        f"{S2}/author/search",
        params={"query": name, "limit": 5, "fields": "name,affiliations,paperCount,hIndex"},
        headers={"User-Agent": "papers-resolver/1.0"},
    )
    r.raise_for_status()
    return r.json().get("data", [])


async def s2_top_papers(client: httpx.AsyncClient, author_id: str) -> List[dict]:
    r = await client.get(
        f"{S2}/author/{author_id}/papers",
        params={"limit": 3, "fields": "title,year,venue,citationCount,url"},
        headers={"User-Agent": "papers-resolver/1.0"},
    )
    r.raise_for_status()
    return r.json().get("data", [])


def score_candidate(name: str, headline: str, cand_name: str, inst: str, extra_boost: float = 0.0) -> float:
    # Simple but effective scoring for V2 baseline:
    # name match (base) + institution overlap with headline tokens + optional boost
    n = (name or "").lower().strip()
    cn = (cand_name or "").lower().strip()
    base = 0.0
    if cn == n:
        base = 0.55
    elif n and (n in cn or cn in n):
        base = 0.35
    else:
        base = 0.20

    ht = tokenize(headline or "")
    it = tokenize(inst or "")
    ov = overlap(ht[:20], it[:40])
    inst_score = 0.0
    if ov >= 2:
        inst_score = 0.25
    elif ov == 1:
        inst_score = 0.12

    return clamp(base + inst_score + extra_boost, 0.0, 0.95)


def make_inmail_line(name: str, headline: str, top_title: Optional[str]) -> str:
    first = (name or "there").split(" ")[0]
    if top_title:
        hook = f'Saw your paper "{top_title}" — especially relevant given your work in {headline or "this space"}.'
    else:
        hook = f"Came across your background in {headline or 'this space'}."
    pitch = "I’m partnering with a high-performing team and your profile looks unusually aligned."
    close = "Open to a quick chat this week?"
    return f"Hi {first} — {hook} {pitch} {close}"


@app.post("/resolve")
async def resolve(req: ResolveRequest):
    async with httpx.AsyncClient(timeout=15) as client:
        # Run searches in parallel
        oa_task = openalex_author_search(client, req.name)
        s2_task = s2_author_search(client, req.name)

        openalex_authors, s2_authors = await oa_task, await s2_task

        candidates = []

        for a in openalex_authors:
            inst = (a.get("last_known_institution") or {}).get("display_name") or ""
            sc = score_candidate(req.name, req.headline or "", a.get("display_name") or "", inst, extra_boost=0.02)
            candidates.append({
                "source": "OpenAlex",
                "id": a.get("id"),
                "name": a.get("display_name"),
                "institution": inst,
                "score": sc
            })

        for a in s2_authors:
            inst = ", ".join(a.get("affiliations") or [])
            # small boost if they have many papers / h-index
            h = float(a.get("hIndex") or 0)
            pc = float(a.get("paperCount") or 0)
            boost = 0.0
            if h >= 15: boost += 0.06
            elif h >= 8: boost += 0.03
            if pc >= 30: boost += 0.03
            sc = score_candidate(req.name, req.headline or "", a.get("name") or "", inst, extra_boost=boost)
            candidates.append({
                "source": "SemanticScholar",
                "id": a.get("authorId"),
                "name": a.get("name"),
                "institution": inst,
                "score": sc
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0] if candidates else None

        if not best:
            return {
                "ok": True,
                "best": None,
                "candidates": [],
                "papers": [],
                "inmail_line": make_inmail_line(req.name, req.headline or "", None),
            }

        papers = []
        if best["source"] == "OpenAlex":
            works = await openalex_top_works(client, best["id"])
            for w in works:
                papers.append({
                    "title": w.get("title"),
                    "year": w.get("publication_year"),
                    "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name")
                            or ((w.get("host_venue") or {}).get("display_name") or ""),
                    "citations": w.get("cited_by_count") or 0,
                    "url": (w.get("primary_location") or {}).get("landing_page_url") or w.get("id"),
                })
        else:
            works = await s2_top_papers(client, str(best["id"]))
            for w in works:
                papers.append({
                    "title": w.get("title"),
                    "year": w.get("year"),
                    "venue": w.get("venue") or "",
                    "citations": w.get("citationCount") or 0,
                    "url": w.get("url"),
                })

        top_title = papers[0]["title"] if papers else None
        inmail = make_inmail_line(req.name, req.headline or "", top_title)

        return {
            "ok": True,
            "best": {
                **best,
                "confidence_label": label(best["score"])
            },
            "candidates": candidates[:5],
            "papers": papers,
            "inmail_line": inmail,
        }


@app.get("/health")
def health():
    return {"ok": True}