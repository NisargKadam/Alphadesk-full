# Retrieval golden set

`golden_questions.json` pairs questions with the source and page that answers
them — the ground truth `python -m app.eval_retrieval` scores against. It
covers the synthetic SampleCo report, so it is safe to commit.

Add a few pairs for your own filings after ingesting them: real documents ask
harder questions than synthetic ones, and your eval is only as honest as its
golden set.
