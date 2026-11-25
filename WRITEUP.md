# Intent Qualification – Writeup

## 0. How I approached the problem

Before writing any pipeline code I spent most of my time understanding the data and the queries:

1. **Data exploration** (477 companies, 13 fields): how often each field is missing, what format each field really has, which countries and industries appear.
2. **Query decomposition**: I split each of the 12 queries into constraints and marked each one as either a **filter** (a clear field and condition, e.g. `country == "ro"`) or an **interpretation** (something that has to be understood, e.g. "logistics company").
3. **Manual labelling before building anything**: I labelled three queries of different difficulty by hand, so I could later measure the system instead of judging it by "looks right":
   - query 1 – *Logistic companies in Romania* (all 26 Romanian companies),
   - query 6 – *Pharmaceutical companies in Switzerland* (all 43 Swiss companies),
   - query 12 – *Critical components for EV battery production* (69 battery-related candidates).

The question I kept coming back to was: **does this company *do* the activity, *supply* it, or just *mention* it?** Almost every hard case in this dataset is a version of that question.

## 1. Approach – architecture

```
companies.jsonl
   │
   ▼
1. load_data()   clean mixed formats, extract country / NAICS / industry
   │
   ▼
2. prefilter()   cheap per-query filter: country, NAICS prefix, numeric thresholds, keywords
   │             (missing numeric values are kept, not dropped)
   ▼
3. qualify()     LLM judges the remaining candidates in batches (≤ 60 per call),
   │             using a per-query "guidance" = my interpretation of the query
   │             + results cached on disk
   ▼
4. rank()        qualified first; companies missing a field the query filters on go lower
   │
   ▼
results/query_N.csv  (+ evaluate.py compares 3 queries with my labels)
```

**Why this design**

- **Cheap filters first.** When I decomposed the queries, every one of them had some structured part (country, size, year, revenue, an industry code) and one part that needed interpretation. The structured part can be done for free with pandas, so there is no reason to pay an LLM to check that a company is in Romania. For some queries the NAICS code already does almost all the work: NAICS 311/312 (food / beverage manufacturing) found exactly the 20 French food and beverage producers, and NAICS 3254 found 27 of my 28 Swiss pharma companies.
- **LLM only for the role question.** NAICS and keywords cannot tell a logistics company from a company that sells forklifts to logistics companies. The LLM can, so it is used only on the candidates that survive the filter (552 candidates over 12 queries instead of 12 × 477).
- **Batching and caching.** One company per call used up the free daily quota (20 requests) after 12 companies. Sending up to 60 companies per call reduced the whole run to **15 calls** (about 12 minutes), and the cache means a query is never paid for twice.
- **Per-query guidance written by me.** Without it the model's interpretation swung wildly (see section 4). The guidance is the written form of the decisions I made during query decomposition, e.g. for query 1: *freight, courier/postal, warehousing services for other companies = YES; wholesalers moving their own goods, fuel companies, port/rail infrastructure managers, sellers of trucks = NO*.
- **Robustness.** The API was often overloaded (HTTP 503), so calls are retried with increasing waits and fall back to other models; when the daily quota is used up the run moves to the next model, and finished queries stay in the cache.

## 2. Tradeoffs

**Embeddings vs. LLM.** I tested sentence embeddings (`all-MiniLM-L6-v2`) on query 1 against my labels:

| Text given to the embedding model | correct companies in the top 6 |
|---|---|
| "Logistic companies in Romania" | 1 of 6 (the port was 18th of 26) |
| "logistics company providing transport, freight and warehousing services for other businesses" | 2 of 6 (but the forklift maker rose to 5th) |

Two lessons: the words "in Romania" matched every company (they are all Romanian), so the country belongs in a filter, not in the embedding; and embeddings rank by **topic**, not by **role** – a forklift maker and a pharma wholesaler with its own trucks look like logistics. The scores of right and wrong answers were also very close, so there was no usable threshold. The LLM, tested on the same hard cases (forklift maker, oil company with a "logistics network", consumer-goods wholesaler, food wholesaler), agreed with my labels in 5 of 6 cases.

So I **did not use embeddings in the final pipeline**: after the filters there are at most ~100 candidates per query, which the LLM can judge directly. Embeddings become useful at larger scale (section 5).

**What I optimised for:** accuracy on role-based queries and very low cost (everything runs on a free tier).

**What I gave up intentionally:**
- **Full automation of query understanding.** The filters and the guidance are written by me for these 12 queries. A new query needs new guidance. I chose this because I could see exactly where the model's interpretation went wrong, and a human-written interpretation was the most reliable fix in the time I had.
- **Speed.** One LLM call takes 10–60 seconds; a pure filter/embedding system would answer instantly.
- **Strict handling of missing data.** Companies without `employee_count`, `year_founded` or `revenue` are kept and ranked lower, instead of being dropped. A plain filter on these columns silently removes them (`NaN > 1000` is `False`), and `employee_count` is missing for 39% of the companies.

## 3. Evaluation

Results of `evaluate.py` against my manual labels (borderline companies are not counted):

| Query | Found | Missed | Wrong | Precision | Recall |
|---|---|---|---|---|---|
| 1. Logistic companies in Romania | 1 | 0 | 0 | 100% | 100% |
| 6. Pharmaceutical companies in Switzerland | 27 | 1 | 1 | 96% | 96% |
| 12. Critical components for EV battery production | 42 | 1 | 0 | 100% | 98% |

For comparison, NAICS 3254 alone on query 6 would find 27 of 28 correct companies but also 5 wrong ones (a nutrition company, a diagnostics company, aesthetic fillers, supplements and a hyaluronic-acid products company). On query 12 NAICS helps much less, because many battery materials producers are classified as chemical manufacturers, not battery manufacturers.

**These numbers need an honest caveat.** The guidance in the prompt and the labels both come from me, so the system is evaluated against the same rules it was built with. That is the reason the scores are high. On a new query without human-written guidance I expect clearly worse results – I saw this happen during development (section 4). Three labelled queries are also a small sample.

Query 1 is a good example of how ambiguous the task is: after applying my own rule consistently ("logistics for other companies"), only **one** of the 26 Romanian companies (Poșta Română) is a clear YES. Two more (Romgaz, Valbur) are borderline, and the port, the railway infrastructure company and the logistics real-estate company turned out to be infrastructure managers or landlords rather than logistics providers.

## 4. Error analysis

**Disagreements in the final run** – all three are borderline cases where the model's argument is defensible:

- **Lonza Capsules & Health Ingredients** (my label YES, system NO): the model read it as a supplier of empty capsule shells to pharma companies. That is exactly my own "supplier for the industry" rule, so the model may be right and my label too generous.
- **TRB Chemedica** (my label NO, system YES): I saw mainly hyaluronic-acid products close to medical devices; the model picked up that it also manufactures active ingredients and prescription medicines.
- **ReFire_ev** (my label YES, system NO): the company mainly builds EV charging infrastructure and distributes batteries; the model judged it does not manufacture battery components.

**Errors I found during development:**

- **Missing input, not wrong reasoning – CFR.** The model first said NO to the Romanian railway company because its description only talks about rail infrastructure; "freight rail transport" appeared only in `core_offerings`, which the model did not receive. I added `core_offerings` (and country, year, employees) to the prompt.
- **Prompt sensitivity.** One rewording of the general rules made the model answer NO to all 26 Romanian companies – even the postal service, "not a third-party logistics provider". The next rewording made it answer YES to 11, including every trap I had identified (a real-estate broker, two wholesalers, a gas transmission operator, a fuel distributor). Adding the per-query guidance fixed this swing.
- **Inconsistency between calls.** The same model judged the port of Constanța YES when asked about it alone, and NO inside a batch of 26.
- **Weaker fallback model.** On a day the main model was overloaded, part of the answers came from a smaller model, which was clearly more permissive. The cache now records which model answered each query.
- **My own labels.** When I compared my query 1 labels with the model's answers, I found that I had been inconsistent (I said YES to the port and the railway infrastructure company, but NO to a forklift maker for the same "supplier/infrastructure" reason). I corrected my labels; this showed me that the ground truth itself is subjective on borderline companies.

## 5. Scaling to 100,000 companies per query

- **Filters stay cheap.** Country, NAICS and numeric filters are simple column operations; at this scale they would run on a database with indexes instead of pandas.
- **Embeddings become useful here.** They were not good enough to *decide*, but they are good enough to *shortlist*: after the filters, rank the remaining thousands of companies by embedding similarity and send only the top few hundred to the LLM. Company embeddings are computed once and stored (a vector index), so each query costs one embedding.
- **LLM cost stays bounded.** Batching, caching per (company, query) and running batches in parallel on a paid tier.
- **Automatic guidance.** Instead of writing the guidance by hand, one LLM call per query could generate the filters and the interpretation – one call per query, not per company – reviewed by a human for important queries.
- **Tiered model use.** A small cheap model for clear cases and the larger model only for uncertain ones.

## 6. Failure modes – confident but wrong

- **Wrong data is trusted.** Walmart's revenue in the dataset is about 2,500 billion $ (real: about 700 billion); ENERCON is described as "Swedish" but is German; Rompetrol and CIRKEL Energi appear twice. The system has no way to detect this.
- **Missing data.** A company without `year_founded` can still be returned for "founded after 2018" (ranked lower). Of the 49 companies returned for query 8, 20 have no founding year and 25 no employee count.
- **Similarity traps.** Words like "logistics", "distribution", "battery" or "e-commerce" appear in companies that are not matches (a forklift maker, an oil company's own supply chain, solar installers, a warehouse landlord for e-commerce).
- **Confident explanations.** The LLM always gives a fluent one-line reason, even when the description is vague; the answer can change with batch size or prompt wording.
- **Constraints the data cannot answer.** "Fast-growing" and "using Shopify" have no field in the data; the system approximates them (small direct-to-consumer brands for Shopify, growth signals in the description) and can be confidently wrong.

**What I would monitor in production:**
- the share of YES answers per query – a sudden 0% or 100% means a prompt or model problem (this is exactly how I noticed the run with 11 YES),
- agreement with a small labelled set, re-checked whenever the prompt or the model changes,
- companies the LLM skipped in a batch, and which model answered,
- how often top results have missing data for the query's key fields.

## 7. Limits and next steps

- **Works well:** queries that map to an industry code (food and beverage in France, pharma in Switzerland) and role-based queries where the guidance is clear (battery components, logistics).
- **Struggles:** queries whose key constraint is not in the data ("fast-growing", "using Shopify") – only 3 companies were returned for query 10, all small French beverage brands, which is an approximation I cannot verify.
- **Assumptions:** "in country X" means headquarters (`country_code`), not where the company operates (Skanska is Swedish but operates mainly in the US); Scandinavia is Sweden, Norway and Denmark, with Finland and Iceland accepted as nearby.
- **Signals relied on most:** NAICS code, description and core offerings. They are misleading when the description is generic or wrong.
- **Next steps:** automatic query decomposition and guidance; embedding shortlist for scale; more labelled queries, labelled by more than one person; structured (JSON) model output instead of parsing text lines; a second LLM pass only for low-confidence answers.

## How to run

```bash
pip install -r requirements.txt
# put GEMINI_API_KEY=... in a .env file (not committed)
python solution.py      # writes results/query_1.csv ... query_12.csv
python evaluate.py      # compares queries 1, 6, 12 with my labels
```

The `cache/` folder contains the LLM answers of the final run, so `python solution.py` reproduces the results without new API calls.

## Note on AI assistance

I used an AI assistant during this project as a tutor and for help with code and English writing. The data exploration, the query decomposition, the manual labels, the experiments and the decisions described above were done by me; the assistant helped me turn them into code and into this text.
