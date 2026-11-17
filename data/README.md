# Data

Comma-separated files, UTF-8, one header line. Nothing here identifies a rater, and no file holds transcript
text. Recordings carry their identifier in the HIMR Public Corpus.

## The questions

`questions.csv`: the sixteen rating questions.

| Column | Meaning |
|---|---|
| `question` | Identifier used in the other files |
| `name` | Short name used in the report |
| `dimension` | The dimension of the analysis that the question marks, F1 to F8 |
| `sign` | +1 or -1: the sign with which the question enters the composite of its dimension |
| `form_study1`, `place_in_study1` | Form (A or B) in the first study and the question's place in that study's list; empty for the four questions asked in the second study only. The place is needed to reproduce the first study's resampling exactly |
| `form_study2`, `place_on_form_study2` | Form in the second study and place on that form, 1 to 8. The order shown to a rater rotates with the rater's set |
| `new_in_study2` | 1 if the question was asked in the second study only |
| `asked`, `level0` to `level4` | The question and its five levels as shown to raters and as the models receive them |
| `plain_sentence_shown_to_raters` | One explanatory sentence shown to raters under four questions; the models do not get it |

## The pool of candidate excerpts and the models' scores

`pool.csv`: the 4,902 eligible candidate excerpts from which both studies drew. A candidate is the first 250
words of a random 500-word passage of a recording.

| Column | Meaning |
|---|---|
| `candidate` | Label of the candidate |
| `recording` | The recording's identifier in the HIMR Public Corpus, release `release_6f67c2b3f2604d7b5587a0ba`. Its page on the site is `/corpus/videos/` followed by the identifier with a hyphen in place of the underscore |
| `first_word`, `last_word` | Where the excerpt lies in the recording's transcript: words counted from 0 in the transcript text split at whitespace; `last_word` is the first word after the excerpt |
| `kind` | `video` or `stream` |
| `period` | 2015-2018, 2019-2021 or 2022-2025 |
| `passage` | 1, 2 or 3: the first, second or third random passage of the recording in the pool |
| `weight` | Weight for figures of the population: every recording counts once and the two kinds count equally |
| `jev_<question>`, `drex_<question>` | Each model's score of the excerpt on the question, 0 to 4: the mean of the model's distribution over the five levels |
| `jev_F1` to `jev_F8`, `drex_F1` to `drex_F8` | Each model's dimension scores of the excerpt, computed as in the analysis from its ratings of 45 questions; the second model is standardised within itself |
| `site_F1` to `site_F8` | The published dimension scores of the recording, from all its passages |
| `d12u_<question>`, `d12u_F1` to `d12u_F8` | The same scores from a third model, rated after both studies; its dimension scores are computed like the second model's |

`jev` is the model `jev-1.14.0`, `drex` is `drex-max`, `d12u` is `d1.2-ultra`.

## First study

`study1_excerpts.csv`: the 160 excerpts.

| Column | Meaning |
|---|---|
| `excerpt` | Label, E001 to E160 |
| `candidate` | Its row in `pool.csv` |
| `part` | `random`: one of the 56 excerpts drawn at random. `chosen`: one of the 104 chosen by rule on the models' scores |
| `kind`, `period` | As in the pool |
| `year` | Date of the recording as a year with a fraction |

`study1_ratings.csv`: 9,840 answers of the 205 raters whose ratings count.

| Column | Meaning |
|---|---|
| `rater` | A001 to A205, in a random order |
| `form`, `set` | The form (A or B) and the set of eight excerpts that the rater read. Sets carry random labels, 0 to 19 |
| `excerpt`, `question` | What was rated |
| `rating` | 0 to 4; empty: the rater chose "cannot judge" |

## Second study

`study2_excerpts.csv`: the 160 excerpts, two passages from each of 80 recordings.

| Column | Meaning |
|---|---|
| `excerpt` | Label, N001 to N160 |
| `candidate` | Its row in `pool.csv`; the recording is given there |
| `passage` | 1 or 2: which of the two passages of the recording |
| `kind`, `period`, `year` | As above |

`study2_sets.csv`: the design. Per form 100 sets of eight excerpts in five rounds of 20; in a round every
excerpt is in exactly one set.

| Column | Meaning |
|---|---|
| `form`, `set` | The form and the set's label, 0 to 99. The labels are random: the page filled sets in order of arrival, so the set's own number would say when a rater took part |
| `round` | 1 to 5, random labels as well |
| `arm` | `sample answers`: raters of this set were shown sample answers after the two practice excerpts. `own answers only`: they were shown their own answers only |
| `first_question` | 1 to 8: the place on the form of the question shown first; the others follow in the form's order |
| `excerpt` | One of the eight excerpts of the set |

`study2_raters.csv`: the 211 raters whose ratings count.

| Column | Meaning |
|---|---|
| `rater` | B001 to B211, in a random order |
| `form`, `set` | The form and the set that the rater read |
| `arm` | The arm of the set |

`study2_time_histogram.csv` and `study2_time_summary.csv`: time on the page from start to finish, as the number
of raters in steps of five minutes by arm (under 5 and over 95 minutes are counted in the first and last step),
and as mean, standard deviation, median, quartiles and the share over 25 minutes. No rater's own time is given.

`study2_ratings.csv`: 13,504 answers.

| Column | Meaning |
|---|---|
| `rater`, `excerpt`, `question` | Who rated what |
| `rating` | 0 to 4; empty: "cannot judge" |
| `position` | 1 to 8: the place of the excerpt in the rater's session |

`study2_model_calls.csv`: four calls of each of the three models on each of the 160 excerpts of the second study: the scores
on the sixteen questions and the eight dimension scores. Call 1 is the call whose scores are in `pool.csv`.

## What was left out, and what remains possible

- Raters: no participant code of the platform, no time of day, no order of arrival, no listing a rater came
  from, no comment, no time per page or per answer, and no rater's own time on the page. Raters are numbered at
  random, apart for each study, and sets carry random labels. Nobody took part in both studies. No demographic
  data were collected.
- Only raters whose ratings count are here: not the two pilots of ten, not participants whom the checks
  excluded, not sessions left unfinished.
- Excerpts: no text, and no mark of which excerpts hold sensitive language. Recordings are identified and the
  excerpt's place in the transcript is given, so the text can be read in the public corpus.
