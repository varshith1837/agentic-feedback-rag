# Assumptions, provenance, and audit notes

Source specification: `Original_Project_Description.docx`, sections 4–17. Input ZIP: `data/brazil.zip` supplied with the request.

## Choices deliberately made explicit

1. **Money:** Sum each order's Olist payments, not payment rows after the item join. Seller/category slices sum each item's `price + freight_value` and are association proxies, not proven seller fault.
2. **Review identity:** Use `review_id::order_id`; duplicate rows for the same pair keep the longer nonempty comment (stable first tie), preventing duplicate risk.
3. **Review population:** All reviews with title or message text. Stratify on **review score and order purchase month**; the reference plan suggested weighting by score alone, but this extension maintains monthly coverage.
4. **Weights:** For each score/month stratum, assign population-written-review count divided by sampled count. Use `MAX(weight)` if more than one sampled issue review from the same order, which avoids monetary duplication but is not an unbiased order-level inclusion-probability estimator.
5. **Fallback:** Invalid extraction after retry becomes `fallback` (aspect `other` and star-derived sentiment for audit) and is **excluded from aggregated risk**. Nonempty valid LLM annotations may contain no aspects.
6. **Demo:** Conservative Portuguese keyword rules are clearly marked `rules_demo`; they can miss paraphrases, sarcasm, mixed sentiment and positive mentions. Never cite their accuracy without hand labels.
7. **Taxonomy:** Eleven fixed labels from the specification; free-text new issue themes and mobile-app feedback are not in-scope.
8. **NLI:** English translation can be wrong; NLI is optional and checks only proposed free-text English review paraphrases. It is not applied to numeric totals, for which exact SQL provenance is stronger evidence.
9. **Scope:** Dates refer to order **purchase date**, not review creation date or physical delivery date; seller state is not customer state.
10. **Results:** Historical dataset is from 2016–2018. Risk is **historical exposure associated with complaints**, not sales leakage, expected future lost sales, causation, or churn.

## Tests and what remains unverified

Local offline environment supported SQLite, httpx, Pydantic, NetworkX and pytest. Validated ingestion of the original ZIP, SQLite database integrity, 30 predefined questions, HTTP protocol behavior via mocks, and regression tests. This environment does not have Streamlit, DuckDB, Ollama, or live Groq connectivity installed, so live deployment and LLM behavior require running the provided commands on the recipient machine. If those external dependencies change, adapt configurations rather than making unsupported zero-error claims.

## Suggested genuine study

- Annotate 60 reviews independently; calculate aspect precision/recall and translation-fidelity checks.
- Stratify a representative sample of 10–15 answers for manual citation correctness and claim faithfulness.
- Compare an **actually generative** answer baseline with checker on/off on the same inputs; report false accept and false reject rates.
- Evaluate order-level weighting via known inclusion probabilities, and assess uncertainty intervals before interpreting weighted estimates.
- Audit disparate labeling across Portuguese variants, star scores and months.
