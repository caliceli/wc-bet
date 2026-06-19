#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World Cup 2026 prediction-pool betting bot  (hybrid: AI scout + math)
=====================================================================
Site : https://prediction-romania-worldcup.lovable.app
Plays the per-match SCORE bets for one participant (default: "flav").

Talks straight to the site's server-function API -- no browser needed.

WHAT IT DOES, each run:
  1. Pulls the full fixture list from the site.
  2. Re-rates every team from the tournament's OWN finished results (Elo with a
     goal-difference multiplier) -- the baseline / sanity check.
  3. For each of the next N fixtures (default 5) that are still open and not yet
     bet, asks an OpenRouter model WITH WEB SEARCH to research the two teams
     (form, injuries, line-ups, head-to-head, stakes) and return win/draw/loss
     probabilities + expected goals.
  4. Blends the AI's expected goals with the Elo baseline, then picks the
     scoreline with the highest EXPECTED points under the pool rule
     +3 exact / +1 correct result / 0 wrong  (Poisson model).
  5. Places those bets (only with --live).

If OPENROUTER_API_KEY is not set, it silently falls back to Elo-only, so it
always works.

USAGE
    python worldcup_bot.py                 # DRY RUN - prints picks, places nothing
    python worldcup_bot.py --live          # place the next 5
ENV
    WC_PLAYER            participant name (default "flav")
    WC_COUNT             games per run (default 5)
    OPENROUTER_API_KEY   enables the AI scout (create at openrouter.ai)
    WC_MODEL             OpenRouter model (default "perplexity/sonar", searches the web natively;
                         for a general model add the suffix, e.g. "openai/gpt-4o:online")
"""

import os, sys, json, math, argparse
from datetime import datetime, timezone, timedelta
import urllib.request, urllib.error

BASE = "https://prediction-romania-worldcup.lovable.app"

SF = {
    "schedule":    "/_serverFn/c433284fb1832f2b84fdb292c47f877af8bdd48d728907c43143db0e38e099be",
    "leaderboard": "/_serverFn/19789f8dbe67051dd8a33c1d4cb8d49da143b9560988fc96ec588304a32ec066",
    "my_bets":     "/_serverFn/2f4bfb675c9e20ac0233df07d95eff1d1730fbb37044fda7dbc6e3e9b9922150",
    "place_bet":   "/_serverFn/8bd6a7e365cf87a05002a1dd8ca50148964691faebdc1ec3c48c508a45aa34a7",
}
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "x-tsr-serverfn": "true",                 # <-- without this the API returns nothing
    "origin": BASE,
    "referer": BASE + "/schedule",
    "user-agent": "Mozilla/5.0 (wc-bot)",
}

LOCK_MARGIN_MIN = 70
MAX_GOALS = 7
DEFAULT_RATING = 1650

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
WC_MODEL = os.environ.get("WC_MODEL", "perplexity/sonar")
LLM_WEIGHT = 0.6     # how much to trust the AI's expected goals vs the Elo baseline (0..1)

# ---------------------------------------------------------------------------
# seroval encode / decode
# ---------------------------------------------------------------------------
def _enc_val(v):
    if isinstance(v, bool):
        return {"t": 2, "s": 3 if v else 4}
    if isinstance(v, int):
        return {"t": 0, "s": v}
    return {"t": 1, "s": str(v)}

def encode(data: dict) -> bytes:
    inner = {"t": 10, "i": 1,
             "p": {"k": list(data.keys()), "v": [_enc_val(v) for v in data.values()]}, "o": 0}
    payload = {"t": {"t": 10, "i": 0, "p": {"k": ["data"], "v": [inner]}, "o": 0}, "f": 63, "m": []}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")

def decode(node):
    if node is None:
        return None
    t = node.get("t")
    if t == 0:  return int(node["s"])
    if t == 1:  return str(node["s"])
    if t == 2:  return {3: True, 4: False}.get(node.get("s"))
    if t == 9:  return [decode(x) for x in node.get("a", [])]
    if t in (10, 11):
        o, p = {}, node.get("p")
        if p:
            for k, v in zip(p["k"], p["v"]):
                o[k] = decode(v)
        return o
    return node.get("s")

# ---------------------------------------------------------------------------
# Site API
# ---------------------------------------------------------------------------
def _call(path, body=None):
    req = urllib.request.Request(BASE + path, data=body,
                                 method="POST" if body is not None else "GET", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} calling {path}: {e.read().decode('utf-8','ignore')[:300]}")
    if not raw:
        raise SystemExit(f"Empty response from {path} (check the x-tsr-serverfn header).")
    return decode(json.loads(raw)).get("result")

def get_schedule():     return _call(SF["schedule"])
def get_leaderboard():  return _call(SF["leaderboard"])
def get_user_bets(uid): return _call(SF["my_bets"], encode({"userId": uid}))
def place_bet(uid, match, h, a):
    return _call(SF["place_bet"], encode({
        "userId": uid, "matchId": match["id"], "matchUtcDate": match["utcDate"],
        "homeScore": h, "awayScore": a}))

def resolve_user_id(name):
    for p in get_leaderboard():
        if str(p.get("name", "")).strip().lower() == name.strip().lower():
            return p["userId"]
    raise SystemExit(f'Player "{name}" not found on the leaderboard.')

# ---------------------------------------------------------------------------
# Elo baseline (seed ratings -> adjusted by tournament results)
# ---------------------------------------------------------------------------
RATINGS = {
    "Argentina": 2105, "France": 2080, "Spain": 2075, "Brazil": 2045,
    "England": 2025, "Portugal": 2010, "Netherlands": 1985, "Germany": 1960,
    "Belgium": 1945, "Uruguay": 1915, "Croatia": 1900, "Colombia": 1900,
    "Morocco": 1885, "Japan": 1830, "Senegal": 1825, "Turkiye": 1825,
    "Austria": 1815, "Switzerland": 1815, "United States": 1805, "Ecuador": 1800,
    "Mexico": 1800, "Korea Republic": 1790, "Norway": 1790, "Czechia": 1760,
    "Iran": 1755, "Algeria": 1745, "Canada": 1745, "Sweden": 1740,
    "Ivory Coast": 1735, "Scotland": 1735, "Egypt": 1730, "Australia": 1725,
    "Bosnia and Herzegovina": 1715, "Paraguay": 1715, "DR Congo": 1700,
    "Ghana": 1700, "South Africa": 1700, "Tunisia": 1685, "Qatar": 1670,
    "Saudi Arabia": 1665, "Uzbekistan": 1660, "Panama": 1655, "Jordan": 1635,
    "Cape Verde": 1635, "Iraq": 1620, "New Zealand": 1595, "Curaçao": 1590,
    "Haiti": 1560,
}

def current_ratings(schedule, K=40):
    R = dict(RATINGS)
    for m in sorted((m for m in schedule if m["status"] == "FINISHED"), key=lambda x: x["utcDate"]):
        h, a = m["homeTeam"], m["awayTeam"]
        if "TBD" in (h, a):
            continue
        rh, ra = R.get(h, DEFAULT_RATING), R.get(a, DEFAULT_RATING)
        exp_home = 1 / (1 + 10 ** (-(rh - ra) / 400))
        hs, as_ = m["homeScore"], m["awayScore"]
        actual = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        gd = abs(hs - as_)
        mult = 1.0 if gd <= 1 else (1.5 if gd == 2 else 1.75 + (gd - 3) / 8)
        delta = K * mult * (actual - exp_home)
        R[h], R[a] = rh + delta, ra - delta
    return R

def recent_form(team, schedule, limit=3):
    games = sorted((m for m in schedule if m["status"] == "FINISHED"
                    and team in (m["homeTeam"], m["awayTeam"])), key=lambda x: x["utcDate"])
    out = []
    for m in games[-limit:]:
        if m["homeTeam"] == team:
            opp, gf, ga = m["awayTeam"], m["homeScore"], m["awayScore"]
        else:
            opp, gf, ga = m["homeTeam"], m["awayScore"], m["homeScore"]
        res = "W" if gf > ga else ("D" if gf == ga else "L")
        out.append(f"{res} {gf}-{ga} v {opp}")
    return "; ".join(out) if out else "no games yet"

# ---------------------------------------------------------------------------
# Poisson + expected-points optimiser
# ---------------------------------------------------------------------------
def _poisson(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)

def _lambdas_from_elo(rh, ra):
    sup = (rh - ra) / 170.0
    total = 2.6
    return max(0.15, (total + sup) / 2), max(0.15, (total - sup) / 2)

def best_score(lh, la):
    """Scoreline maximising expected points under +3 exact / +1 result."""
    P = [[_poisson(i, lh) * _poisson(j, la) for j in range(MAX_GOALS + 1)]
         for i in range(MAX_GOALS + 1)]
    best = None
    for a in range(MAX_GOALS + 1):
        for b in range(MAX_GOALS + 1):
            ep = 0.0
            for i in range(MAX_GOALS + 1):
                for j in range(MAX_GOALS + 1):
                    p = P[i][j]
                    if a == i and b == j:
                        ep += 3 * p
                    elif (a - b) * (i - j) > 0 or (a - b == 0 and i - j == 0):
                        ep += 1 * p
            if best is None or ep > best[2]:
                best = (a, b, ep)
    return best

# ---------------------------------------------------------------------------
# AI scout (OpenRouter, web search)
# ---------------------------------------------------------------------------
def _extract_json(s):
    i, j = s.find("{"), s.rfind("}")
    return json.loads(s[i:j + 1])

def llm_scout(home, away, context):
    """Research the match via an OpenRouter model with web search.
    Returns {p_home,p_draw,p_away,exp_home,exp_away,note} or None on failure."""
    if not OPENROUTER_KEY:
        return None
    prompt = (
        f"You are a football analyst predicting the 2026 FIFA World Cup match {home} vs {away}. "
        f"Search the web for the latest on both teams: current form, injuries and suspensions, "
        f"likely line-ups, head-to-head history, and what each side needs from this match. "
        f"{context} "
        "Then output ONLY a JSON object (no prose, no markdown) with keys: "
        "p_home, p_draw, p_away (probabilities the HOME team wins/draws/loses, summing to ~1), "
        "exp_home, exp_away (expected goals for each team, realistic, 0-5), "
        "note (one short sentence naming the decisive factor)."
    )
    body = json.dumps({
        "model": WC_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body, method="POST",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json",
                 "HTTP-Referer": BASE, "X-Title": "WC bot"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
        obj = _extract_json(data["choices"][0]["message"]["content"])
        for k in ("p_home", "p_draw", "p_away", "exp_home", "exp_away"):
            obj[k] = float(obj[k])
        if not (0 <= obj["exp_home"] <= 6 and 0 <= obj["exp_away"] <= 6):
            return None
        return obj
    except Exception as e:
        print(f"  (AI scout unavailable for {home} v {away}: {e}; using Elo only)")
        return None

def predict_hybrid(home, away, ratings, schedule):
    """Blend the AI's expected goals with the Elo baseline, then optimise the scoreline."""
    lh_e, la_e = _lambdas_from_elo(ratings.get(home, DEFAULT_RATING),
                                   ratings.get(away, DEFAULT_RATING))
    ctx = (f"Their results so far -- {home}: {recent_form(home, schedule)}. "
           f"{away}: {recent_form(away, schedule)}.")
    scout = llm_scout(home, away, ctx)
    if scout:
        lh = LLM_WEIGHT * scout["exp_home"] + (1 - LLM_WEIGHT) * lh_e
        la = LLM_WEIGHT * scout["exp_away"] + (1 - LLM_WEIGHT) * la_e
        a, b, ep = best_score(lh, la)
        return a, b, ep, "AI+Elo", scout.get("note", "")
    a, b, ep = best_score(lh_e, la_e)
    return a, b, ep, "Elo", ""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))

def main():
    ap = argparse.ArgumentParser(description="World Cup 2026 betting bot")
    ap.add_argument("--player", default=os.environ.get("WC_PLAYER", "flav"))
    ap.add_argument("--live", action="store_true", help="actually place bets")
    ap.add_argument("--count", type=int, default=int(os.environ.get("WC_COUNT", "5")),
                    help="max games to bet per run (default 5)")
    args = ap.parse_args()

    live = args.live and os.environ.get("WC_DRYRUN", "") != "1"
    brain = f"AI scout ({WC_MODEL}) + Elo" if OPENROUTER_KEY else "Elo only (set OPENROUTER_API_KEY for AI)"
    print(f"=== WC bot — player: {args.player} — mode: {'LIVE' if live else 'DRY RUN'} — brain: {brain} ===")

    uid = resolve_user_id(args.player)
    print(f"resolved userId = {uid}")

    schedule = get_schedule()
    ratings = current_ratings(schedule)
    n_finished = sum(1 for m in schedule if m["status"] == "FINISHED")
    already = {b["matchId"] for b in get_user_bets(uid)}
    cutoff = datetime.now(timezone.utc) + timedelta(minutes=LOCK_MARGIN_MIN)

    candidates = [
        m for m in sorted(schedule, key=lambda x: x["utcDate"])
        if m["status"] == "TIMED" and "TBD" not in (m["homeTeam"], m["awayTeam"])
        and m["id"] not in already and parse_dt(m["utcDate"]) > cutoff
    ][: args.count]

    print(f"ratings adjusted from {n_finished} finished matches\n")
    placed = 0
    for m in candidates:
        h, a, ep, src, note = predict_hybrid(m["homeTeam"], m["awayTeam"], ratings, schedule)
        line = f"{m['utcDate']}  {m['homeTeam']} vs {m['awayTeam']}  ->  {h}-{a}  [{src}, E[pts]={ep:.2f}]"
        print(line)
        if note:
            print(f"      {note}")
        if live:
            place_bet(uid, m, h, a)
            placed += 1
    if live:
        print(f"\nPlaced {placed} bet(s) for {args.player}.")
    else:
        print(f"\nDry run: would place {len(candidates)} bet(s). Add --live to commit.")

if __name__ == "__main__":
    main()
