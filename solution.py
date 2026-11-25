"""
Intent Qualification - Veridion challenge

Pipeline (one pass per query):
    1. load_data()      - load companies.jsonl and normalise messy fields
    2. QUERIES          - each query = text + a cheap structured pre-filter
    3. prefilter()      - apply the cheap filter (country, NAICS, numbers, keywords)
    4. qualify()        - LLM judges the remaining candidates in batches, results cached on disk
    5. rank()           - qualified companies first; companies with missing data for
                          the query's numeric constraints are ranked lower, not dropped
    6. main()           - runs all 12 queries and writes results/ (one CSV per query)

Design idea: cheap filters remove the obvious non-matches for free, and the
expensive LLM call is only used where real judgement is needed (the company's
ROLE: does it do the activity, supply it, or just mention it?).
"""

import ast
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

DATA_PATH = "companies.jsonl"
CACHE_DIR = "cache"
RESULTS_DIR = "results"
MODEL = "gemini-3.5-flash"
# used in order when the main model is overloaded (503); unknown names are skipped
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3-flash", "gemini-2.5-flash"]
BATCH_SIZE = 60          # companies per LLM call (free tier = 20 calls/day -> 15 calls for all 12 queries)
PAUSE_SECONDS = 5        # pause between LLM calls (rate limit)


# ---------------------------------------------------------------------------
# 1. Data loading and cleaning
# ---------------------------------------------------------------------------

def to_dict(x):
    """address / primary_naics are mixed: 413 rows are dicts, 64 are dict-like strings."""
    if isinstance(x, str):
        return ast.literal_eval(x)   # json.loads fails: the strings use single quotes
    return x


def load_data(path=DATA_PATH):
    df = pd.read_json(path, lines=True)
    df["address"] = df.address.apply(to_dict)
    df["primary_naics"] = df.primary_naics.apply(to_dict)

    df["country"] = df.address.str.get("country_code")
    df["naics"] = df.primary_naics.str.get("code").astype(str)
    df["industry"] = df.primary_naics.str.get("label")

    # one lowercase text field used by the keyword pre-filters
    df["text"] = (df.description.fillna("") + " " + df.core_offerings.astype(str)).str.lower()
    return df


# ---------------------------------------------------------------------------
# 2. Queries: text + cheap pre-filter
#    The pre-filters come from my manual query decomposition (Etapa 2).
#    Rule for numbers: missing values are KEPT (and ranked lower), never dropped,
#    because a filter on NaN silently removes ~40% of companies.
# ---------------------------------------------------------------------------

EUROPE = ["at", "be", "ch", "de", "dk", "es", "fi", "fr", "gb", "gr", "hr", "ie", "is",
          "it", "lt", "lu", "nl", "no", "pl", "pt", "ro", "ru", "se", "tr", "ua"]
NORDIC = ["se", "no", "dk", "fi", "is"]   # strict Scandinavia = se, no, dk


def gt_or_missing(col, value):
    return (col > value) | col.isna()


def lt_or_missing(col, value):
    return (col < value) | col.isna()


QUERIES = [
    {
        "id": 1,
        "text": "Logistic companies in Romania",
        "filter": lambda d: d.country == "ro",
        "guidance": "YES: companies that provide freight transport, courier/postal or warehousing services to other companies. NO: infrastructure managers (port authorities, railway infrastructure managers), landlords of logistics real estate, wholesalers/distributors moving their own goods, fuel/energy companies, sellers of trucks or forklifts, gas transmission operators.",
        "numeric": [],
    },
    {
        "id": 2,
        "text": "Public software companies with more than 1,000 employees",
        "filter": lambda d: d.is_public
        & gt_or_missing(d.employee_count, 1000)
        & (d.naics.str.startswith(("5112", "5132", "5415", "5182", "5191"))
           | d.text.str.contains("software")),
        "guidance": "Software company = its main business is creating software, either as products/platforms or as custom software development (IT services firms like TCS or Capgemini count). NO: business process outsourcing / call centers, pure management consulting, financial firms.",
        "numeric": ["employee_count"],
    },
    {
        "id": 3,
        "text": "Food and beverage manufacturers in France",
        # NAICS 311 = food, 312 = beverage; 3122 = tobacco is excluded
        "filter": lambda d: (d.country == "fr")
        & d.naics.str.startswith(("311", "312"))
        & ~d.naics.str.startswith("3122"),
        "guidance": "YES: companies that produce food or beverages (incl. wine, spirits, soft drinks). NO: tobacco, packaging makers, wholesalers/retailers, associations or networks.",
        "numeric": [],
    },
    {
        "id": 4,
        "text": "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
        "filter": lambda d: d.naics.str.startswith(("3261", "3262", "3272", "3221", "3222"))
        | d.text.str.contains("packaging"),
        "guidance": "YES: manufacturers or distributors of packaging (bottles, jars, tubes, boxes...) that mention cosmetics, beauty or personal care among their markets. NO: cosmetics brands themselves (they are the customer, not the supplier), packaging makers with no link to cosmetics.",
        "numeric": [],
    },
    {
        "id": 5,
        "text": "Construction companies in the United States with revenue over $50 million",
        "filter": lambda d: (d.country == "us")
        & gt_or_missing(d.revenue, 50_000_000)
        & d.naics.str.startswith("23"),
        "guidance": "YES: construction contractors (buildings, infrastructure, specialty construction). NO: suppliers of building materials or construction equipment, real estate developers that do not build.",
        "numeric": ["revenue"],
    },
    {
        "id": 6,
        "text": "Pharmaceutical companies in Switzerland",
        # all Swiss companies: "companies" is wider than NAICS 3254 (e.g. DNDi is outside it)
        "filter": lambda d: d.country == "ch",
        "guidance": "YES: companies that develop or produce medicines, including contract manufacturers (CDMO) of drug ingredients and non-profit drug developers. NO: nutrition/supplements, diagnostics, aesthetic medical devices, contract research organizations that only run clinical trials.",
        "numeric": [],
    },
    {
        "id": 7,
        "text": "B2B SaaS companies providing HR solutions in Europe",
        "filter": lambda d: d.country.isin(EUROPE)
        & (d.text.str.contains("human resource|payroll|recruit|talent| hr ")
           | d.naics.str.startswith("541612")),
        "guidance": "HR solutions include HR software, recruitment, payroll. The company must offer software as a service (SaaS) sold to businesses. NO: pure HR consulting without a software product.",
        "numeric": [],
    },
    {
        "id": 8,
        "text": "Clean energy startups founded after 2018 with fewer than 200 employees",
        "filter": lambda d: gt_or_missing(d.year_founded, 2018)
        & lt_or_missing(d.employee_count, 200)
        & d.text.str.contains("solar|wind|batter|renewable|clean energy|electric vehicle"
                              "|energy storage|hydrogen"),
        "guidance": "Clean energy = solar, wind, batteries and energy storage, electric vehicles, hydrogen. NO: energy-efficiency consulting, fossil fuels. A startup should be developing a new product or technology.",
        "numeric": ["year_founded", "employee_count"],
    },
    {
        "id": 9,
        "text": "Fast-growing fintech companies competing with traditional banks in Europe",
        "filter": lambda d: d.country.isin(EUROPE)
        & (d.naics.str.startswith("52")
           | d.text.str.contains("fintech|payment|banking|neobank|lending")),
        "guidance": "YES: companies that offer banking-like services directly to customers (payments, accounts, cards, loans, investing). NO: companies that sell software or infrastructure TO banks (suppliers, not competitors), traditional banks themselves. Mention in the reason if there are signs of fast growth.",
        "numeric": [],
    },
    {
        "id": 10,
        "text": "E-commerce companies using Shopify or similar platforms",
        "filter": lambda d: d.business_model.astype(str).str.lower().str.contains("e-commerce")
        | d.text.str.contains("e-commerce|ecommerce|online store|shopify"),
        "guidance": "YES: small or medium brands whose main business is selling their own products online, directly to consumers (the typical Shopify user). NO: large retailers or manufacturers with their own in-house online stores (e.g. Walmart, Dell), companies that only provide services to e-commerce (warehousing, logistics, agencies).",
        "numeric": [],
    },
    {
        "id": 11,
        "text": "Renewable energy equipment manufacturers in Scandinavia",
        "filter": lambda d: d.country.isin(NORDIC)
        & d.text.str.contains("solar|wind|turbine|renewable|batter|energy storage|hydro"),
        "guidance": "YES: manufacturers of complete renewable energy equipment (wind turbines, solar panels, inverters, energy storage systems). NO: developers/operators of wind or solar parks, installers, component-only suppliers, gas or steam turbine makers.",
        "numeric": [],
    },
    {
        "id": 12,
        "text": "Companies that manufacture or supply critical components for electric vehicle battery production",
        "filter": lambda d: d.text.str.contains("batter|lithium|cathode|anode|separator"
                                                 "|electrolyte|cobalt|graphite|nickel")
        | d.naics.str.startswith("3359"),
        "guidance": "YES: battery cell manufacturers, producers of battery components or materials (cathode, anode, separator, electrolyte) and suppliers of critical raw materials (lithium, nickel, cobalt, graphite). NO: makers of equipment for battery factories, companies that only install or use batteries (e.g. solar installers), technology/IP licensing companies.",
        "numeric": [],
    },
]


def prefilter(df, query):
    return df[query["filter"](df)].copy()


# ---------------------------------------------------------------------------
# 3. LLM qualification (batched + cached)
# ---------------------------------------------------------------------------

RULES = """Rules:
- The interpretation above has priority. A company is YES if its core business belongs to the requested category, even if it does not use the exact words of the query. Mentioning the domain in passing is not enough.
- Companies that sell equipment or services FOR the requested domain (suppliers), or that buy from it (customers), are NO - unless the query itself asks for suppliers.
- Companies that do the activity only for their own products (e.g. they transport their own goods) are NO.
- Check every condition in the query (country, industry, size, year...). If the data contradicts a condition, answer NO. If a value is missing, judge only on the rest."""


def describe(row, i):
    return (f"[{i}] {row.operational_name} | country: {row.country} | founded: {row.year_founded} "
            f"| employees: {row.employee_count} | public: {row.is_public}\n"
            f"Description: {row.description}\nOfferings: {row.core_offerings}")


def build_prompt(query, batch):
    companies = "\n\n".join(describe(row, i) for i, row in batch.iterrows())
    return f"""You check whether each company in the list matches a user's search query.

Query: {query["text"]}

How to interpret this query (decided by a human analyst):
{query["guidance"]}

{RULES}

Companies:
{companies}

Answer with exactly one line per company, in this format:
[number] YES or NO | reason in one sentence
Write nothing else."""


def parse_answer(text):
    results = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("[") and "]" in line:
            try:
                idx = int(line[1:line.index("]")])
            except ValueError:
                continue
            results[idx] = line[line.index("]") + 1:].strip()
    return results


def call_llm(client, prompt, retries=4):
    """Tries the main model, then the fallback models.
    - 503/500 (overloaded) or 429 per minute: wait and retry, then move to the next model
    - 404 (model name not available): skip to the next model
    - 429 per DAY: stop - finished queries stay in the cache, rerun tomorrow"""
    last_error = None
    for model in [MODEL] + FALLBACK_MODELS:
        for attempt in range(1, retries + 1):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text, model
            except genai_errors.APIError as e:
                last_error = e
                if e.code == 429 and "per day" in str(e).lower():
                    raise
                if e.code == 404:
                    break                      # this model name does not exist -> next model
                if e.code not in (429, 500, 503):
                    raise
                if attempt == retries:
                    print(f"  {model} still busy, switching model")
                    break
                wait = 15 * attempt
                print(f"  {model} busy ({e.code}), retrying in {wait}s ({attempt}/{retries})")
                time.sleep(wait)
    raise last_error


def qualify(client, query, candidates):
    """Returns {index: "YES | reason"}; cached per query so reruns cost nothing."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"query_{query['id']}.json")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cached = json.load(f)
        return {int(k): v for k, v in cached["answers"].items()}

    answers = {}
    models_used = set()
    for start in range(0, len(candidates), BATCH_SIZE):
        batch = candidates.iloc[start:start + BATCH_SIZE]
        text, model = call_llm(client, build_prompt(query, batch))
        answers.update(parse_answer(text))
        models_used.add(model)
        time.sleep(PAUSE_SECONDS)

    missing = set(candidates.index) - set(answers)
    if missing:
        print(f"  warning: LLM skipped {len(missing)} companies: {sorted(missing)}")

    print(f"  answered by: {', '.join(sorted(models_used))}")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"models": sorted(models_used), "answers": answers}, f, ensure_ascii=False, indent=2)
    return answers


# ---------------------------------------------------------------------------
# 4. Ranking and output
# ---------------------------------------------------------------------------

def rank(candidates, answers, query):
    out = candidates[["operational_name", "website", "country", "industry"]].copy()
    out["llm_answer"] = [answers.get(i, "") for i in out.index]
    out["qualified"] = out.llm_answer.str.upper().str.startswith("YES")
    out["reason"] = out.llm_answer.str.split("|", n=1).str[-1].str.strip()

    # companies missing a field the query filters on are kept, but ranked lower
    out["missing_data"] = False
    for col in query["numeric"]:
        out["missing_data"] |= candidates[col].isna()

    out = out.sort_values(["qualified", "missing_data"], ascending=[False, True])
    return out.drop(columns="llm_answer")


def run_query(client, df, query):
    candidates = prefilter(df, query)
    answers = qualify(client, query, candidates)
    return rank(candidates, answers, query)


def main():
    load_dotenv()                 # GEMINI_API_KEY is read from .env (never committed)
    client = genai.Client()
    df = load_data()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for query in QUERIES:
        result = run_query(client, df, query)
        qualified = result[result.qualified]
        result.to_csv(os.path.join(RESULTS_DIR, f"query_{query['id']}.csv"))
        print(f"Query {query['id']}: {query['text']}")
        print(f"  candidates after pre-filter: {len(result)} | qualified: {len(qualified)}")
        for i, row in qualified.head(10).iterrows():
            flag = " (missing data)" if row.missing_data else ""
            print(f"    [{i}] {row.operational_name}{flag}")


if __name__ == "__main__":
    main()
