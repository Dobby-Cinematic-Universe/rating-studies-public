# Do people read transcripts as the models do?

Two studies in which people rated excerpts of machine transcripts that two language models had rated, with the
data and the code behind every number.

- `report/report.pdf`: the report (35 pages). Its LaTeX source is beside it.
- `data/`: every rating of every rater, without anything that identifies a rater, and the models' scores.
  `data/README.md` describes every file and column.
- `replicate.py`: recomputes every figure of the report from `data/`.
- `results/`: what `replicate.py` writes.
- `expected.json`: the numbers of the report, for `replicate.py --check`.

## Running it

```sh
python3 replicate.py --check
```

Needs Python 3.9 or later with numpy and scipy, and takes a few minutes. Every random draw is seeded, so the
numbers come out the same on every run; `--check` compares them with `expected.json` and reports any that
differ. `--boot 200` uses fewer resamples and is faster; the intervals then differ from the report's.

## What the results are

Ratings of the wording of 250-word excerpts by crowd raters and by two language models. They are not facts about
any person's mind, motives or private life, and an excerpt can hold more than one voice. The report says what was
fixed before the ratings were collected and what was computed afterwards.

## What is not here

No excerpt text. Every excerpt is given by its recording in the HIMR Public Corpus and its word positions, so
the text can be looked up there by those who want to read it.

Nothing that identifies a participant: no participant code, no time of day, no order of arrival, no comment, no
time per answer and no rater's own time on the page. No ratings of pilot participants, of participants whose
ratings did not count, or of sessions left unfinished. Section 7 of the report and `data/README.md` give the
details and name the three results of the report that cannot be recomputed from these files.
