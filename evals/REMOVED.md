# Fixtures removed from `critic_fixtures.jsonl`, and why

A fixture that leaves the set silently takes its reason with it, and the next
person re-derives it — or, worse, rebuilds the same bad fixture. Every removal
gets an entry here: what it was, why it went, and what replaced it.

---

## `eval-20260820-162253-marketing-translation-fa-en::omission::1::fd9a5d5e`

**Removed:** 2026-08-20 · **Was labelled:** `revise`, `borderline`, `tuned`
**Replaced by:** `eval-20260820-172405-marketing-translation-fa-en::omission::2::3ca88033`

**What it was.** A mutation of the NetBan translation output in which the source
claim *"so your work continues without interruption"* was replaced by the vaguer
*"keeping everything running smoothly"*. It was meant to test whether the Critic
notices a source claim that has been softened rather than deleted.

**Why it was removed.** The label did not survive an independent check.

The fixture was missed ten times running — across two Gemini models, with and
without the Critic's per-item counting rule, and both before and after the
criterion was split from one five-part list into five atomic criteria. That
consistency was read as a stable failure of the judge for three tuning rounds
before anyone questioned the label itself.

It was then shown to `grok-4.6` — this project's Manager, which neither authored
the fixture (Claude did) nor graded it (Gemini did) — with the text and the single
criterion and no hint of the expected answer. Three runs returned **NO / YES /
YES**, all three quoting the same phrase.

So two independent vendors read the text as satisfying the criterion more often
than not, and one of them disagreed with itself. The criterion is **under-determined
by the text**: *"keeping everything running smoothly"* neither clearly asserts nor
clearly omits that the user's work continues without interruption.

A fixture whose expected answer three vendors cannot agree on measures neither the
judge's accuracy nor its failure. Every rate it appears in is contaminated by it.

**Why it was deleted rather than relabelled `accept`.** Flipping the label would be
the same mistake pointing the other way. We do not know the text satisfies the
criterion; we know it is **not judgeable**. Neither label is defensible.

**What replaced it.** The same criterion, tested with a mutation that deletes the
claim outright instead of softening it — no substitute phrase at all. Before being
added, it went through the same blind check: `grok-4.6` answered **NO three times
out of three**, each time noting the text never mentions the user's work. The check
ran *before* admission, which is the order this one should have followed.

**The general rule this produced** (also in §10 of the handoff report): when a
fixture misses several runs in a row, test its label with a vendor that neither
built nor graded it, and say nothing about what you expect — before you spend a
round tuning a prompt to close a gap that may not exist.
