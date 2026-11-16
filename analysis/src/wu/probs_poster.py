# Copyright (c) 2026 Daria Bikina
# SPDX-License-Identifier: MIT

"""Plain-language figures for the token-probability experiment (P1 in PLAN.md), for the poster canvas.

    wu probs-report --model qwen3-8b --figures-only   ->   <RUNS>/<model>/probs-<tag>-plain-{1..4}-*.png

Drawn from probs-<tag>-report.json; nothing is computed here except unit changes. Each figure has one
claim, English and Russian side by side, one row per compression method, and says what the model is
asked, what is read from it and what is counted. AUROCs are shown as "pairs ordered correctly, of 100",
which is the same number times 100. `example_card` writes one real question out in full for the canvas.
The detailed figures with every definition are the ones `probs_report.figures` writes.
"""

import numpy as np

from .confidence_report import LABELS, _colour, _key

LANGUAGE = {"en": "English", "ru": "Russian"}
MIN_LOST = 20  # rows with fewer lost facts are left empty in the kept-against-lost figure
FAMILIES = "Blue: quantization. Orange: WANDA pruning. Grey: magnitude pruning and block removal."
INTERVALS = "Lines through the dots: 95% bootstrap intervals over facts."
ASKED = (
    "What is asked: {n} factual questions that each have one correct answer, for example 'Who is the author of "
    "The Bicentennial Man?' (Isaac Asimov). Each is asked in two wordings, in English and in Russian."
)
KNOWS_A_FACT = (
    "For each question we read the probability the model gives the correct answer and each wrong answer on a "
    "fixed list of plausible ones (here, other authors). The model knows the fact if the correct answer gets "
    "the highest probability."
)
P_YES = (
    "P(Yes): the probability that the model replies Yes to 'Do you know the answer to this question? Who is the "
    "author of The Bicentennial Man? Answer with just one word, Yes or No.'"
)


def _dot(ax, e, y, setting, style="filled", scale=100.0):
    if e is None or np.isnan(e["est"]):
        return
    colour = _colour(setting)
    ax.plot([scale * e["lo"], scale * e["hi"]], [y, y], color=colour, lw=2.5, alpha=0.45, zorder=2)
    ax.plot(
        scale * e["est"], y, "o", ms=13, mew=2.5, color="white" if style == "hollow" else colour, mec=colour, zorder=3
    )


def _panels(fig, rect, rows, top_gap=0.0):
    """Two axes side by side, English and Russian, with the row labels to the left of the first."""
    h = fig.get_figheight()
    bottom, top = rect[1] + 0.95 / h, rect[3] - top_gap - 0.55 / h
    axes = {}
    for lang, x in (("en", 0.25), ("ru", 0.64)):
        ax = fig.add_axes([x, bottom, 0.33, top - bottom])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(LANGUAGE[lang], fontsize=21, loc="left", color="0.2", pad=14)
        axes[lang] = ax
    side = axes["en"].get_yaxis_transform()
    for i, s in enumerate(rows):
        axes["en"].text(
            -0.06, len(rows) - 1 - i, LABELS[s], ha="right", va="center", fontsize=18, color=_colour(s), transform=side
        )
    return axes


def _pairs_axis(ax, label, low=40):
    from .gt.poster import SHADE

    ax.axvline(50, color="0.3", lw=1.2)
    ax.axvspan(0, 50, color=SHADE, zorder=0, lw=0)
    drawn = [x for line in ax.lines for x in line.get_xdata() if np.isfinite(x)]
    ax.set_xlim(min([low, *(x - 3 for x in drawn)]), 100)  # never cut an interval
    ax.set_xticks([50, 60, 70, 80, 90, 100], ["50\nchance", "60", "70", "80", "90", "100\nall"])
    ax.set_xlabel(label)


def _same_scale(axes):
    """Panels that are compared share one x range."""
    low = min(ax.get_xlim()[0] for ax in axes.values())
    for ax in axes.values():
        ax.set_xlim(low, 100)


def _figure(plt, k_rows: int, read: list[str], n_key: int = 0):
    """A figure tall enough for its reading text, its key and k_rows rows."""
    import textwrap

    lines = sum(len(textwrap.wrap(t, int(15 * 0.96 * 72 / (0.47 * 19)))) for t in read)
    return plt.figure(figsize=(15, 4.0 + 0.37 * lines + 0.55 * n_key + 0.62 * k_rows))


def figures(report: dict, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .gt.poster import frame, save, theme

    theme()
    model, tag, S = report["model"], report["tag"], report["settings"]
    name = "Qwen3-" + model.split("-")[1].upper()
    rows = list(S)
    comp = [s for s in rows if s != "base"]
    if not comp:  # only the uncompressed model has been run (e.g. the smoke run): nothing to compare
        return
    R = {lang: {s: S[s][lang] for s in S} for lang in LANGUAGE}
    asked = ASKED.format(n=report["n_facts"])
    pct = lambda v: f"{100 * v:.0f}"
    flip = lambda e: {"est": 1 - e["est"], "lo": 1 - e["hi"], "hi": 1 - e["lo"]}
    span = lambda key, among: " and ".join(
        f"{pct(min(R[lang][s][key]['est'] for s in among))}–{pct(max(R[lang][s][key]['est'] for s in among))} "
        f"in {LANGUAGE[lang]}"
        for lang in LANGUAGE
    )

    # 1. how many facts are lost
    lost = {lang: {s: flip(R[lang][s]["retention"]) for s in comp} for lang in LANGUAGE}
    most = {lang: max(comp, key=lambda s: lost[lang][s]["est"]) for lang in LANGUAGE}
    read = [
        asked,
        f"What is measured: {KNOWS_A_FACT[0].lower()}{KNOWS_A_FACT[1:]} The uncompressed model knows "
        f"{pct(R['en']['base']['first']['est'])}% of the facts in English and "
        f"{pct(R['ru']['base']['first']['est'])}% in Russian.",
        "Bars: of the facts the uncompressed model knew, the percentage each compressed model no longer knows.",
    ]
    fig = _figure(plt, len(comp), read)
    rect = frame(
        fig,
        f"Compression makes {name} lose up to {pct(lost['en'][most['en']]['est'])}% of the facts it knew in "
        f"English and {pct(lost['ru'][most['ru']]['est'])}% in Russian",
        read,
        f"{name}. Thin lines: 95% bootstrap intervals over facts. {FAMILIES}",
    )
    axes = _panels(fig, rect, comp)
    for lang, ax in axes.items():
        for i, s in enumerate(comp):
            y, e = len(comp) - 1 - i, lost[lang][s]
            ax.barh(y, 100 * e["est"], height=0.62, color=_colour(s), lw=0, zorder=2)
            ax.plot([100 * e["lo"], 100 * e["hi"]], [y, y], color="0.2", lw=1.5, zorder=3)
            ax.text(100 * e["hi"] + 1.5, y, f"{pct(e['est'])}%", va="center", fontsize=17, color="0.2")
        ax.set_xlim(0, 70)
        ax.set_xticks([0, 25, 50], ["0", "25%", "50%"])
        ax.set_xlabel("known facts now lost")
    save(fig, out / f"probs-{tag}-plain-1-forgetting.png")

    # 2. which number goes with a correct answer
    claim = all(
        min(R[lang][s]["margin_for_correct"]["lo"] for s in rows)
        > max(R[lang][s]["knows_for_correct"]["est"] for s in rows)
        for lang in LANGUAGE
    )
    read = [
        asked + " The model writes its answer in a few words, and the answer is marked correct or wrong.",
        "Two numbers are read for each question, defined under ○ and ● below. What is measured: among all pairs "
        "of one correctly answered and one wrongly answered question, the number of pairs out of 100 in which "
        "the number is higher for the correctly answered question. Chance gives 50.",
        f"Result: of 100 pairs, the answer margin orders {span('margin_for_correct', rows)} correctly, with or "
        f"without compression. P(Yes) orders {span('knows_for_correct', rows)} correctly.",
    ]
    key = [
        (
            "○",
            "Answer margin: how much more probable the model finds the correct answer than the most probable wrong "
            "answer on a fixed list of plausible ones.",
        ),
        ("●", P_YES),
    ]
    fig = _figure(plt, len(rows), read, 4)
    rect = frame(
        fig,
        "A model's answer probabilities show which questions it answers correctly, before and after compression. "
        "Its reply to 'Do you know the answer?' shows this less well"
        if claim
        else f"{name}: which of two numbers goes with a correct answer, before and after compression",
        read,
        f"{name}. {INTERVALS} {FAMILIES}",
    )
    h = fig.get_figheight()
    used = _key(fig, rect[3] - 0.05 / h, key)
    axes = _panels(fig, rect, rows, top_gap=used + 0.1 / h)
    for ax in axes.values():
        for i, s in enumerate(rows):
            y = len(rows) - 1 - i
            lang = next(k for k, v in axes.items() if v is ax)
            _dot(ax, R[lang][s]["margin_for_correct"], y + 0.14, s, "hollow")
            _dot(ax, R[lang][s]["knows_for_correct"], y - 0.14, s)
        _pairs_axis(ax, "pairs ordered correctly, of 100")
    _same_scale(axes)
    save(fig, out / f"probs-{tag}-plain-2-shows-vs-says.png")

    # 3. does P(Yes) follow what was lost
    shown = {lang: [s for s in comp if R[lang][s]["n_lost"] >= MIN_LOST] for lang in LANGUAGE}
    d = {(lang, s): R[lang][s]["d_kept_vs_lost"] for lang in LANGUAGE for s in shown[lang]}
    up = [f"{LABELS[s]} in {LANGUAGE[lang]} (+{pct(v['est'])} pairs)" for (lang, s), v in d.items() if v["lo"] > 0]
    down = [f"{LABELS[s]} in {LANGUAGE[lang]} (−{pct(-v['est'])} pairs)" for (lang, s), v in d.items() if v["hi"] < 0]
    exceptions = (f" ● is higher in: {'; '.join(up)}." if up else "") + (
        f" ● is lower in: {'; '.join(down)}." if down else ""
    )
    read = [
        "A model knows a fact, such as who wrote The Bicentennial Man, if it gives the correct answer a higher "
        "probability than every wrong answer on a fixed list of plausible ones. For each compression method, the "
        "facts the uncompressed model knew are split into those the compressed model still knows (kept) and "
        "those it no longer knows (lost).",
        "What is measured: among all pairs of one kept and one lost fact, the number of pairs out of 100 in which "
        "P(Yes) is higher for the kept fact. Chance gives 50.",
        "○ is above 50 because the facts that get lost are ones the uncompressed model was already less sure "
        "of. A compressed model that registered its losses would lower P(Yes) on the lost facts, which would "
        "put ● well to the right of ○.",
        f"Result: ● and ○ do not differ (the 95% interval of their difference includes zero) in "
        f"{len(d) - len(up) - len(down)} of {len(d)} cases.{exceptions}",
    ]
    key = [
        ("○", "Before compression: the uncompressed model's P(Yes), on the facts this method will keep or lose."),
        ("●", "After compression: the compressed model's P(Yes), on the same facts."),
        ("", P_YES),
    ]
    fig = _figure(plt, len(comp), read, 4)
    rect = frame(
        fig,
        "Asking a compressed model 'Do you know the answer?' does not pick out the facts it has lost"
        if len(up) <= 2
        else f"{name}: does the reply to 'Do you know the answer?' follow which facts survive compression",
        read,
        f"{name}. Empty rows: fewer than {MIN_LOST} questions lost. {INTERVALS} {FAMILIES}",
    )
    h = fig.get_figheight()
    used = _key(fig, rect[3] - 0.05 / h, key)
    axes = _panels(fig, rect, comp, top_gap=used + 0.1 / h)
    for lang, ax in axes.items():
        for i, s in enumerate(comp):
            y = len(comp) - 1 - i
            if s not in shown[lang]:
                continue
            _dot(ax, R[lang][s]["kept_vs_lost_base"], y + 0.14, s, "hollow")
            _dot(ax, R[lang][s]["kept_vs_lost"], y - 0.14, s)
        _pairs_axis(ax, "kept–lost pairs ordered correctly, of 100")
    _same_scale(axes)
    save(fig, out / f"probs-{tag}-plain-3-kept-vs-lost.png")

    # 4. how often it says Yes, for real subjects and for invented ones
    yes = [R[lang][s]["p_yes"]["est"] for lang in LANGUAGE for s in comp]
    worst = max((R[lang][s]["p_yes_controls"]["est"], s, lang) for lang in LANGUAGE for s in comp)
    else_ = [f"{LABELS[s]} in {LANGUAGE[lang]}" for lang in LANGUAGE for s in rows if R[lang][s]["yes_no_mass"] < 0.6]
    other = f" The model would often reply with something else after: {'; '.join(else_)}." if else_ else ""
    read = [
        f"What is asked: 'Do you know the answer to this question?', followed by a question about a real subject "
        f"({report['n_facts']} of them) or about an invented one (100 of them), for example 'Who is the author of "
        "The Widow of Kessering Street?', a book that does not exist. The model is told to answer Yes or No.",
        "What is measured: the average probability that the model replies Yes. A model that knows what it knows "
        "replies Yes for real subjects (●) and No for invented ones (○).",
        "Result: each compression method moves ● and ○ together, towards Yes or towards No. More or fewer Yes "
        "replies after compression therefore do not mean the model knows more or less. After "
        f"{LABELS[worst[1]]}, the model replies Yes about invented subjects with probability {pct(worst[0])}% in "
        f"{LANGUAGE[worst[2]]}.",
    ]
    fig = _figure(plt, len(rows), read, 2)
    rect = frame(
        fig,
        f"Compression moves how often {name} replies Yes to 'Do you know the answer?' to anywhere between "
        f"{pct(min(yes))}% and {pct(max(yes))}%, and moves it the same way for subjects that do not exist",
        read,
        f"{name}. Each subject is asked about in two wordings. Probability of Yes relative to No.{other} "
        f"Lines through the dots: 95% bootstrap intervals. {FAMILIES}",
    )
    h = fig.get_figheight()
    used = _key(
        fig,
        rect[3] - 0.05 / h,
        [("●", "Questions about real subjects."), ("○", "Questions about invented subjects, which have no answer.")],
    )
    axes = _panels(fig, rect, rows, top_gap=used + 0.1 / h)
    for lang, ax in axes.items():
        for i, s in enumerate(rows):
            y, real, made = len(rows) - 1 - i, R[lang][s]["p_yes"], R[lang][s]["p_yes_controls"]
            ax.plot([100 * made["est"], 100 * real["est"]], [y, y], color="0.8", lw=5, zorder=1, solid_capstyle="round")
            _dot(ax, made, y, s, "hollow")
            _dot(ax, real, y, s)
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 50, 100], ["0\nnever", "50%", "100\nalways"])
        ax.set_xlabel("average probability of replying Yes")
    save(fig, out / f"probs-{tag}-plain-4-yes.png")


def _chance(logp: float) -> str:
    """A probability in words: a percentage, or '1 in N' when it is small."""
    p = float(np.exp(logp))
    if p >= 0.001:
        return f"{100 * p:.0f}%" if p >= 0.095 else f"{100 * p:.1f}%"
    n = 1 / p
    for size, word in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if n >= size:
            return f"1 in {n / size:.0f} {word}"
    return f"1 in {n:.0f}"


def example_card(report: dict) -> str | None:
    """Markdown for the canvas: one real question taken through every step, with the numbers the
    uncompressed model actually gave (report["example"], written by `probs_report.example`)."""
    e = report.get("example")
    if not e:
        return None
    a, w, inv, t = e["correct"], e["wrong"], e["invented"], e["then_correct"]
    en = report["settings"]["base"]["en"]
    assert a["correct"] and not w["correct"] and not t["correct"] and a["margin"] > 0 > t["margin"]
    assert a["p_yes"] > w["p_yes"], "the example pair should be one P(Yes) orders correctly"
    yes = lambda x: f"{x['p_yes']:.2f}"
    return "\n".join(
        [
            "## One question from start to finish (uncompressed Qwen3-8B)",
            "",
            "Every question has one correct answer that we know. Three things are done with it.",
            "",
            "**1. The model answers the question.** After three solved examples that show the format, it gets:",
            "",
            f"> Answer with only this person's name, without explanation. {a['question']}",
            "",
            f"It writes **{a['wrote']}**, which is correct. This question counts as *answered correctly*.",
            "",
            f"**2. We read how probable the model finds each answer on a list.** For this question the list has "
            f"{a['n_candidates']} names: the correct answer in its spellings, and wrong answers chosen to be "
            "plausible (other authors). The model does not write anything here. For each name we compute the "
            "probability that the model would give it as its answer to the prompt above.",
            "",
            "| answer on the list | probability |",
            "|---|---|",
            f"| {a['gold']} (correct) | {_chance(a['gold_logp'])} |",
            f"| {a['best_wrong']} (the most probable wrong answer) | {_chance(a['best_wrong_logp'])} |",
            "",
            "The correct answer is more probable than every wrong one, so the model *knows* this fact. The "
            f"*answer margin* says by how much: ln({np.exp(a['gold_logp']):.2f}) − ln(probability of "
            f"{a['best_wrong']}) = **{a['margin']:.1f}**. A margin below zero means a wrong answer is more "
            "probable than the correct one.",
            "",
            "**3. Separately, we ask the model whether it knows.** In a new conversation with no examples it gets:",
            "",
            f"> Do you know the answer to this question? {a['question']} Answer with just one word, Yes or No.",
            "",
            f"Again it does not write. We read the probability of its next word: Yes {100 * a['p_yes_token']:.0f}%, "
            f"No {100 * a['p_no_token']:.0f}%, anything else "
            f"{100 * (1 - a['p_yes_token'] - a['p_no_token']):.0f}%. Between Yes and No that is "
            f"**P(Yes) = {yes(a)}**.",
            "",
            "**The same three steps for two more questions**",
            "",
            "| question | the model wrote | correct answer | answer margin | P(Yes) |",
            "|---|---|---|---|---|",
            f"| {a['question']} | {a['wrote']} (correct) | {a['gold']} | +{a['margin']:.1f} | {yes(a)} |",
            f"| {w['question']} | {w['wrote']} (wrong) | {w['gold']} | −{-w['margin']:.1f} | {yes(w)} |",
            f"| {inv['question']} (an invented book) | {inv['wrote']} | none exists | not defined | {yes(inv)} |",
            "",
            "**How the figures below use these numbers**",
            "",
            f"- *Does P(Yes) go with a correct answer?* Pair a correctly answered question with a wrongly "
            f"answered one, such as the first two rows. P(Yes) is {yes(a)} against {yes(w)}: higher for the "
            "correctly answered one, so this pair is ordered correctly. Over all such pairs in English, P(Yes) "
            f"orders {100 * en['knows_for_correct']['est']:.0f} of 100 correctly and the answer margin "
            f"{100 * en['margin_for_correct']['est']:.0f}. Chance gives 50.",
            f"- *Does the model say No when there is nothing to know?* For invented subjects the average P(Yes) "
            f"is {en['p_yes_controls']['est']:.2f}; for real ones {en['p_yes']['est']:.2f}.",
            f"- *What happens after compression?* The same model pruned with {e['then']} answers the first "
            f"question with {t['wrote']}. It now gives {t['gold']} a probability of {_chance(t['gold_logp'])}, "
            f"less than {t['best_wrong']} ({_chance(t['best_wrong_logp'])}): the fact is *lost*. Its P(Yes) "
            f"for this question fell from {yes(a)} to {yes(t)}, which looks as if it noticed. But P(Yes) fell "
            f"on the facts it kept as well: from {e['p_yes_lost'][0]:.2f} to {e['p_yes_lost'][1]:.2f} on "
            f"average for lost facts, and from {e['p_yes_kept'][0]:.2f} to {e['p_yes_kept'][1]:.2f} for kept "
            "ones. The third figure tests whether the reply separates the two groups.",
        ]
    )
