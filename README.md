# guide-gap

**English** | [日本語](README.ja.md)

Reads a batch of support tickets and a guide, and answers one question per ticket:
**is the page missing, or does it exist and nobody could find it?**

Those two look identical in a ticket queue and they have different fixes and
different owners.

```
content gap      -> somebody has to write a page
findability gap  -> the page exists; the title, the search terms, or the
                    cross-links from where the user actually was are wrong
not deflectable  -> no document could have prevented this ticket
```

Conflating the first two produces the worst outcome available: new pages covering
ground the guide already covers, which makes the guide bigger, harder to search, and
therefore worse at the thing that caused the ticket. Volume of new pages is not
progress. The third bucket matters as much -- counting access requests and defects
as guide failures makes the self-service target unreachable, and a target nobody can
hit stops being used.

Built on AWS: Bedrock via boto3 (Titan Text Embeddings V2 for retrieval, Claude
Sonnet 5 via the Converse API for the yes/no calls and drafting). No agent
framework -- this is one scheduled pass over a batch, and an agent that cannot show
*why* it grouped two tickets together is worse here than arithmetic that can.

## Run it

```bash
python -m venv .venv && make install
make test                    # 84 tests, no credentials, ~5s
make analyse                 # one pass against Bedrock, writes out/drafts/
make eval                    # 3 repeats, scored against the labels
make eval-baseline           # the similarity-threshold baseline, no model calls
```

`AWS_REGION` defaults to `us-west-2`. Everything measured below was run against a
verification account, and the per-run artifacts are committed under `evals/` so any
number here can be recomputed without spending a call.

## What it does, in order

```
embed each section of each guide page   (not each page -- see below)
  -> shortlist candidate passages per ticket by cosine similarity
  -> ask, per passage: does this let the user decide what to do next?
  -> content gap / findability gap / not deflectable
  -> cluster ONLY the content gaps, by asking "would one page serve both?"
  -> draft ONLY the clusters where more than one person asked
```

Clustering everything and drafting the biggest groups is the obvious shape and it is
wrong. The biggest group in a support queue is usually a question the guide already
answers -- that is *why* it is the biggest group -- so drafting from raw volume
writes a second page about MFA and leaves the real hole untouched.

## Measured

22 tickets, 7 guide pages, 32 sections. Every label was written by hand **before**
any model ran, and scoring is exact string comparison against those labels. No model
grades another model's output anywhere in this repository.

The two error directions are counted separately, because they are not equally bad:

- **wasteful** — said "write a page" when the guide already answered. A reviewer
  sees the duplicate and rejects the draft.
- **costly** — said "the guide covers it" when nothing was written. The gap
  disappears from the report and nobody looks again.

| version | accuracy | wasteful | costly | pages proposed |
|---|---|---|---|---|
| similarity threshold only (baseline) | 17/22 (77%), 3/3 runs identical | 5 | 0 | 2 |
| entailment, passages arriving empty | 14/22 (64%) | 5 | 1 | 0 |
| entailment, `maxTokens=8` | 19/22 (86%) | 2 | 1 | 0 |
| **entailment, `maxTokens=256`** | **20.7/22 mean (91–95%), 3 runs** | 1–2 | **0 across all 3 runs** | 2 |

Raw per-run data: [`evals/results.json`](evals/results.json) (entailment) and
[`evals/results-threshold.json`](evals/results-threshold.json) (baseline).

The middle two rows are in this table on purpose. One of them is worse than the
baseline, and both were caused by bugs that produced confident, stable, plausible
output — which is the failure mode worth documenting, not the one worth hiding.

### The three findings behind that table

**Similarity cannot decide coverage.** Best-section score per ticket, measured:
tickets the guide answers land in 0.241–0.595, tickets it does not land in
0.205–0.370. Three answered tickets sit inside the unanswered range, so any cut
above 0.370 misclassifies t04, t07 and t10, and any cut below 0.241 misclassifies
t11, t14 and t17. No threshold separates them; tuning it only changes which cases
are wrong. Similarity measures topical closeness, and "is the answer in here"
is entailment. A page about billing is topically close to every billing question
ever asked, including the ones it says nothing about. So similarity was demoted to
shortlisting and a yes/no call decides — twice: once for coverage, once for whether
two tickets belong on one page. `COVERED = 0.45` is kept only as the baseline, and
it is *fitted to these 22 tickets* — that is the finding, not a caveat.

**Sections, not pages.** Whole-page embedding puts every ticket in a 0.17–0.61 band
regardless of whether the page answered it: a 400-word procedure page and a
two-sentence question differ mostly in length and register, and that dominates the
cosine. Max-over-sections asks the question that matters — is there a *passage* here
that answers this. Related: the coverage shortlist must **not** be deduplicated to
one entry per page. Doing that means only a page's best-scoring section is ever
examined, so a ticket answered by a *different* section is reported as an unwritten
page. That single mistake caused four of the eight errors in the first entailment
run.

**A truncated reply is not an error.** `maxTokens=8` looked like a cheap correctness
check — the answer is one word. But Claude Sonnet 5 emits a `reasoningContent` block
before its text and that block is charged against `maxTokens`, so the call returned
HTTP 200, a well-formed message, and **no text block at all**. The empty string
parsed as 「いいえ」, so every truncated call became a confident negative — stably,
across three repeats, which is exactly what made it look like real judgement. At 64
tokens, 8 of 64 calls still truncated; observed reasoning length is around 160
tokens, so the cap now sits at 256. `Usage.truncated` counts `stopReason ==
"max_tokens"` and the eval prints it, so this cannot recur silently. Turning
reasoning off entirely (`thinking: disabled`) works and returns an answer in 6
tokens — and measurably changes verdicts for the worse, so the cheap path is
available and not taken.

Two smaller ones, both worth knowing before starting:

- **`temperature` is rejected outright** by Sonnet 5 — `ValidationException:
  temperature is deprecated for this model`. The usual `temperature=0` lever for
  reproducibility does not exist, which is why the eval measures stability by
  repeating the run instead of assuming the output is pinned. It is also why the
  91–95% above is a range and not a number.
- **An empty passage gets a perfectly reasonable 「いいえ」.** When `build_index`
  omitted the passage text, the batch completed, reported every ticket as a content
  gap, recorded zero errors, and looked like a finding. `entail.answers()` now
  raises on an empty passage rather than asking about it.

### Two found by rendering the output for a person

Both of these survived a green test suite, and both were caught while building
[guide-review](https://github.com/lulperle/guide-review), a review screen for
these proposals. They are the argument for putting machine output in front of a reader
before trusting the numbers about it.

- **`Draft.open_questions` was truncating every wrapped item.** It kept only lines
  *beginning* with a bullet, and the drafts wrap at around forty full-width
  characters, so a caveat arrived as 「命名規則があるのかどうかは、」 and nothing after it.
  It went unnoticed because the property is used for its *count*, and the count was
  right — the number of open questions per draft, the figure this project reports as
  evidence that the drafting stage is safe, was correct while the questions themselves
  were cut in half. The first fix required the continuation line to be indented; the
  next run wrapped flush left instead and broke it again, so the rule is now
  CommonMark lazy continuation, with a blank line closing the item.
- **The draft title came out as the literal string 「タイトル」.** The cause was the
  drafting prompt: its output template showed `# タイトル` under an instruction to use
  the headings exactly as written, so the model did precisely that and put the real
  title on the line below. `well_formed` returned `True`, because all three required
  headings were present and the title was never checked. Both the prompt and the check
  are fixed — see `PLACEHOLDER_TITLES` — because a placeholder title is the specific
  shape prompt drift takes here, and `well_formed` exists to catch drift rather than
  to assume it stopped.

### What is still wrong

- **t08 fails every run.** It is a retrieval miss, not a judgement one: the section
  of `account-request` covering 年度末の払い出し期間 never enters the top-5
  shortlist, so the yes/no call never sees it. The match is lexically obvious
  (年度末, 営業日) and dense-only retrieval misses it — the fix is hybrid lexical +
  dense, which is not implemented here.
- **t10 flips between runs**, and so does clustering: t14/t15/t16 came out as one
  cluster of three in run 1 and split 2+1 in runs 2 and 3. Same input, same code.
  With no temperature control, this is the floor on how stable a single run can be,
  and it is the reason every number above is reported with its repeat count.
- **The fixture is invented.** `fixtures/tickets.yaml` is 22 tickets I wrote to
  resemble a government-cloud helpdesk queue. The pipeline is real and the
  measurements are real, but they are measurements against my own idea of what such
  a queue looks like. Real ticket text would change the numbers, and probably not
  upward.
- **Pair confirmation is O(pairs)**, and pairs are quadratic in batch size. Fine at
  22 tickets (9 calls) and manageable at a few hundred; a batch of thousands needs
  blocking or a cheaper first pass. Not solved here.

## Cost of one pass

Measured, not estimated, across five passes: 55–63 coverage checks, 8–10 pair checks,
2 drafts, ~41–47k input tokens, 1.6–4.6k output tokens, 148–165 seconds. The output
spread is 確認が必要な点 — a run where the drafts admit six things each costs roughly
double one where they admit three, which is the correct direction for that cost to
run. Embeddings are cached in
`.cache/embeddings.json`, so the deterministic half of the pipeline re-runs for free
and only the judgement calls cost anything.

## Layout

```
guide_gap/
  bedrock.py    the only file that talks to AWS
  corpus.py     load pages and tickets; split pages into sections
  retrieve.py   cosine search over sections; pure arithmetic, no network
  entail.py     the two yes/no prompts, and the token-budget lesson
  coverage.py   the three-way call, and the order the checks happen in
  cluster.py    single-link union-find over confirmed pairs
  draft.py      drafting, required to list what it does not know
  metrics.py    the report, with both denominators shown
  score.py      deterministic scoring against the labels
  pipeline.py   one pass, wiring the above together
corpus/guide/   7 markdown pages with YAML front matter
fixtures/       22 labelled tickets
evals/          the repeat harness and its committed raw output
infra/          CDK stack -- synthesised and asserted, never deployed
docs/           one unedited generated draft, as evidence
```

Everything above `bedrock.py` is pure functions over vectors and text, which is why
the test suite runs in CI with no credentials at all.

## Not claimed

- The CDK stack has **never been deployed**. It synthesises and
  `tests/test_infra.py` asserts on the template; no resources exist in any account.
  See [`infra/README.md`](infra/README.md) for what a real deployment would still
  need (a VPC with interface endpoints, a customer-managed key, and the ticket
  export wiring).
- No self-service rate improvement is claimed. This measures *where the gaps are*
  and drafts candidate pages; whether publishing them deflects tickets is a
  before/after measurement on a real guide with real traffic, and it has not been
  done.
- The generated drafts are unreviewed machine output. One is committed at
  [`docs/example-draft.md`](docs/example-draft.md) precisely to show what the
  drafting prompt does with something it does not know: it lists the missing tag
  naming rules under 確認が必要な点 rather than inventing them. That is the property
  that makes the stage safe to automate, and `Draft.well_formed` fails the draft if
  that heading is absent. `evals/export_review.py` writes one pass out as a review
  bundle — every verdict with the passage it was made about — for
  [guide-review](https://github.com/lulperle/guide-review), which is where a
  person accepts or rejects them.

## Licence

MIT.
