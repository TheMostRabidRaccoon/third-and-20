#!/usr/bin/env python3
"""
RACCOON PACK v3.0 - Multi-AI Query System with Synthesis Awareness
Updated: January 8, 2026

NEW IN v3.0:
- Loads relevant synthesis JSONs based on query topic
- Injects prior raccoon consensus into context
- Flags contradictions with existing synthesis
- GPT removed (replaced with Grok Heavy + Claude Pro)
"""

import asyncio, json, os, sys, time, re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PATH CONFIGURATION - UPDATE THIS TO YOUR PROJECT LOCATION
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Assumes code/tools/raccoon_pack_v3.py

# Core paths (relative to PROJECT_ROOT)
LOGS_DIR = PROJECT_ROOT / "logs"
OUTPUTS_DIR = PROJECT_ROOT / "ai" / "outputs"
SYNTHESIS_DIR = PROJECT_ROOT / "ai" / "synthesis"
CONTEXT_FILE = PROJECT_ROOT / "boot_context.md"

# Synthesis subdirectories
SYNTHESIS_THIRD_AND_20 = SYNTHESIS_DIR / "third_and_20"
SYNTHESIS_ANANSI = SYNTHESIS_DIR / "anansi"
SYNTHESIS_CROSS_PROJECT = SYNTHESIS_DIR / "cross_project"

# =============================================================================
# API KEYS
# =============================================================================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GEMINI_MODEL = "gemini-2.0-flash"
GROK_MODEL = "grok-3"

# =============================================================================
# CLIENT SINGLETONS
# =============================================================================
_anthropic_client = None
_grok_client = None
_gemini_client = None

def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import AsyncAnthropic
        _anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _anthropic_client

def get_grok_client():
    global _grok_client
    if _grok_client is None:
        from openai import AsyncOpenAI
        _grok_client = AsyncOpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")
    return _grok_client

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

# =============================================================================
# TOPIC DETECTION
# =============================================================================
TOPIC_KEYWORDS = {
    "third_and_20": [
        "third", "20", "football", "sdi", "cv", "computer vision", "hudl", 
        "player", "tracking", "jersey", "ocr", "snap", "recognition", "latency",
        "brush", "ricky", "transfer portal", "athlete", "coach", "film"
    ],
    "anansi": [
        "anansi", "clinical", "medical", "patient", "emr", "epic", "cerner",
        "urgent care", "diagnosis", "billing", "documentation", "discharge",
        "lab", "imaging", "mch", "healthcare"
    ],
    "funding": [
        "funding", "grant", "investor", "revenue", "pathway", "sbir", "third frontier",
        "b2c", "b2b", "subscription", "price", "cost"
    ]
}

def detect_topics(query: str) -> list:
    query_lower = query.lower()
    detected = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            detected.append(topic)
    if detected:
        detected.append("cross_project")
    return detected if detected else ["cross_project"]

# =============================================================================
# SYNTHESIS LOADING
# =============================================================================
def load_synthesis_files(topics: list) -> dict:
    synthesis_data = {}
    topic_dirs = {
        "third_and_20": SYNTHESIS_THIRD_AND_20,
        "anansi": SYNTHESIS_ANANSI,
        "funding": SYNTHESIS_THIRD_AND_20,
        "cross_project": SYNTHESIS_CROSS_PROJECT
    }
    for topic in topics:
        topic_dir = topic_dirs.get(topic)
        if topic_dir and topic_dir.exists():
            for json_file in topic_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        synthesis_data[json_file.stem] = data
                        print(f"  📚 Loaded: {json_file.name}")
                except Exception as e:
                    print(f"  ⚠️ Failed: {json_file.name}: {e}")
    return synthesis_data

def format_synthesis_context(synthesis_data: dict) -> str:
    if not synthesis_data:
        return ""
    parts = ["\n\n## PRIOR RACCOON SYNTHESIS:\n"]
    for name, data in synthesis_data.items():
        parts.append(f"\n### {name.replace('_', ' ').title()}")
        parts.append(f"Updated: {data.get('last_updated', 'Unknown')}")
        if "consensus" in data:
            parts.append(f"Consensus: {json.dumps(data['consensus'], indent=2)}")
        if data.get("contradictions"):
            parts.append(f"⚠️ Contradictions: {data['contradictions']}")
        if data.get("decisions_made"):
            parts.append(f"Decisions: {data['decisions_made']}")
    return "\n".join(parts)

# =============================================================================
# CONTEXT LOADING
# =============================================================================
def load_full_context(query: str) -> str:
    parts = []
    if CONTEXT_FILE.exists():
        try:
            parts.append(CONTEXT_FILE.read_text())
            print(f"  📖 Boot context loaded")
        except Exception as e:
            print(f"  ⚠️ Boot context failed: {e}")
    topics = detect_topics(query)
    print(f"  🎯 Topics: {topics}")
    synthesis_data = load_synthesis_files(topics)
    if synthesis_data:
        parts.append(format_synthesis_context(synthesis_data))
    return "\n\n".join(parts)

# =============================================================================
# QUERY FUNCTIONS
# =============================================================================
async def query_claude(query, context):
    full_query = context + "\n\n---\n\nQUERY:\n" + query
    start = time.time()
    try:
        client = get_anthropic_client()
        response = await client.messages.create(
            model=CLAUDE_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": full_query}]
        )
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens if response.usage else None
        error = None
    except Exception as e:
        content, tokens, error = None, None, str(e)
    return {"model": CLAUDE_MODEL, "raccoon": "Claude", "response": content, 
            "latency": round(time.time() - start, 2), "tokens": tokens, "error": error}

async def query_gemini(query, context):
    full_query = context + "\n\n---\n\nQUERY:\n" + query
    start = time.time()
    try:
        client = get_gemini_client()
        response = await asyncio.to_thread(
            client.models.generate_content, model=GEMINI_MODEL, contents=full_query
        )
        content = response.text
        error = None
    except Exception as e:
        content, error = None, str(e)
    return {"model": GEMINI_MODEL, "raccoon": "Gemini", "response": content,
            "latency": round(time.time() - start, 2), "tokens": None, "error": error}

async def query_grok(query, context):
    full_query = context + "\n\n---\n\nQUERY:\n" + query
    start = time.time()
    try:
        client = get_grok_client()
        response = await client.chat.completions.create(
            model=GROK_MODEL, messages=[{"role": "user", "content": full_query}],
            max_tokens=2000, temperature=0.8
        )
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else None
        error = None
    except Exception as e:
        content, tokens, error = None, None, str(e)
    return {"model": GROK_MODEL, "raccoon": "Grok", "response": content,
            "latency": round(time.time() - start, 2), "tokens": tokens, "error": error}

# =============================================================================
# SYNTHESIS
# =============================================================================
async def synthesize_responses(query, responses, prior_synthesis):
    valid = [r for r in responses if r.get("response")]
    if not valid:
        return "ERROR: No valid responses.", []
    raw = "\n\n---\n\n".join([f"**{r['raccoon']}** ({r['latency']}s):\n{r['response']}" for r in valid])
    prior = f"\n\nPRIOR SYNTHESIS:\n{json.dumps(prior_synthesis, indent=2)}" if prior_synthesis else ""
    
    prompt = f"""Synthesize these responses to: {query}

{raw}{prior}

Identify:
1) AGREEMENT across raccoons
2) DISAGREEMENT between raccoons
3) CONTRADICTIONS with prior synthesis (flag as ⚠️ CONTRADICTION: [description])
4) Best unique insights
5) NEW INFO that should update synthesis files
6) Recommended action"""

    try:
        client = get_anthropic_client()
        resp = await client.messages.create(
            model=CLAUDE_MODEL, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text
        contradictions = re.findall(r'⚠️ CONTRADICTION: (.+?)(?:\n|$)', text)
        return text, contradictions
    except Exception as e:
        return f"SYNTHESIS ERROR: {e}\n\n{raw}", []

# =============================================================================
# OUTPUT
# =============================================================================
def save_outputs(query, responses, synthesis, topics, contradictions):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    
    # JSON log
    log_path = LOGS_DIR / f"raccoon_{ts.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump({"timestamp": ts.isoformat(), "version": "3.0", "query": query,
                   "topics": topics, "responses": responses, "synthesis": synthesis,
                   "contradictions": contradictions}, f, indent=2)
    
    # Markdown summary
    sum_path = OUTPUTS_DIR / f"synthesis_{ts.strftime('%Y%m%d_%H%M%S')}.md"
    with open(sum_path, "w") as f:
        f.write(f"# Raccoon Pack v3.0 Synthesis\n")
        f.write(f"**Generated:** {ts.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Topics:** {', '.join(topics)}\n\n")
        f.write(f"**Query:** {query}\n\n---\n\n{synthesis}\n")
    
    return log_path, sum_path

# =============================================================================
# MAIN
# =============================================================================
async def query_the_pack(query):
    print(f"\n🦝 RACCOON PACK v3.0")
    print(f"📝 Query: {query[:80]}{'...' if len(query) > 80 else ''}\n")
    
    context = load_full_context(query)
    topics = detect_topics(query)
    prior = load_synthesis_files(topics)
    
    print(f"\n⏳ Querying Claude, Gemini, Grok...\n")
    responses = await asyncio.gather(
        query_claude(query, context),
        query_gemini(query, context),
        query_grok(query, context),
        return_exceptions=True
    )
    responses = [r if isinstance(r, dict) else {"raccoon": "?", "error": str(r)} for r in responses]
    
    for r in responses:
        status = "✅" if r.get("response") else "❌"
        err = f" ({r.get('error')})" if r.get('error') else ""
        print(f"  {status} {r.get('raccoon')}: {r.get('latency', '?')}s{err}")
    
    print(f"\n🧠 Synthesizing...")
    synthesis, contradictions = await synthesize_responses(query, responses, prior)
    
    if contradictions:
        print(f"\n⚠️ {len(contradictions)} CONTRADICTION(S):")
        for c in contradictions:
            print(f"   - {c}")
    
    log_path, sum_path = save_outputs(query, responses, synthesis, topics, contradictions)
    print(f"\n📁 Log: {log_path.name}")
    print(f"📄 Summary: {sum_path.name}")
    print(f"\n{'='*60}\nSYNTHESIS:\n{'='*60}\n{synthesis}\n{'='*60}")

def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("🦝 Query: ").strip()
    if query:
        asyncio.run(query_the_pack(query))

if __name__ == "__main__":
    main()
