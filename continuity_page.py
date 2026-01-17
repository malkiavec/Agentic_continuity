# continuity_page.py
# Replit page: Lotto 6/59 + Bonus (treated as equal 7th member)
# - Loads draws (expects draw_date, num1..num6, bonus OR draw_date, winning_numbers, bonus)
# - Computes continuity features + continuity score (regime breaks)
# - Plots continuity score and shows breakpoints
# - Persists draws + features to SQLite for reproducibility

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from typing import List

# ----------------------------
# SQLite persistence
# ----------------------------

DB_PATH = "lotto_continuity.db"

def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Draws table (unchanged)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS draws (
            draw_date TEXT PRIMARY KEY,
            num1 INTEGER, num2 INTEGER, num3 INTEGER, num4 INTEGER, num5 INTEGER, num6 INTEGER,
            bonus INTEGER
        )
    """)

    # Continuity table: add continuity_state + regime_id
    cur.execute("""
        CREATE TABLE IF NOT EXISTS continuity (
            draw_date TEXT PRIMARY KEY,
            sum INTEGER,
            range INTEGER,
            gap_mean REAL,
            gap_std REAL,
            gap_max INTEGER,
            consec_pairs INTEGER,
            odd INTEGER,
            even INTEGER,
            low INTEGER,
            high INTEGER,
            overlap_prev REAL,
            delta_mean_abs REAL,
            delta_max_abs REAL,
            delta_neg REAL,
            delta_zero REAL,
            delta_pos REAL,
            continuity_score REAL,
            continuity_z REAL,
            regime_break INTEGER,
            continuity_state TEXT,
            regime_id INTEGER
        )
    """)

    # Transition log table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS state_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prev_date TEXT,
            date TEXT,
            regime_id INTEGER,
            prev_state TEXT,
            new_state TEXT,
            transition TEXT,
            continuity_score REAL,
            continuity_z REAL
        )
    """)

    # Break explanation table (tidy)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS break_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            break_date TEXT,
            regime_id INTEGER,
            metric TEXT,
            before_mean REAL,
            after_mean REAL,
            delta REAL,
            pct_change REAL,
            continuity_z REAL,
            state_at_break TEXT
        )
    """)

    conn.commit()

def upsert_draws(conn: sqlite3.Connection, df7: pd.DataFrame) -> None:
    # df7 must contain: draw_date, num1..num6, bonus
    df7 = df7.copy()
    df7["draw_date"] = pd.to_datetime(df7["draw_date"]).dt.strftime("%Y-%m-%d")
    cols = ["draw_date","num1","num2","num3","num4","num5","num6","bonus"]
    df7 = df7[cols]

    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO draws (draw_date,num1,num2,num3,num4,num5,num6,bonus)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(draw_date) DO UPDATE SET
          num1=excluded.num1, num2=excluded.num2, num3=excluded.num3, num4=excluded.num4,
          num5=excluded.num5, num6=excluded.num6, bonus=excluded.bonus
    """, df7.values.tolist())
    conn.commit()

def upsert_continuity(conn: sqlite3.Connection, df_out: pd.DataFrame) -> None:
    # df_out must contain draw_date + continuity columns
    df_out = df_out.copy()
    df_out["draw_date"] = pd.to_datetime(df_out["draw_date"]).dt.strftime("%Y-%m-%d")

    insert_cols = [
        "draw_date","sum","range","gap_mean","gap_std","gap_max","consec_pairs",
        "odd","even","low","high",
        "overlap_prev","delta_mean_abs","delta_max_abs","delta_neg","delta_zero","delta_pos",
        "continuity_score","continuity_z","regime_break",
        "continuity_state","regime_id"
    ]
    df_out = df_out[insert_cols].copy()
    df_out["regime_break"] = df_out["regime_break"].fillna(False).astype(bool).astype(int)
    df_out["continuity_state"] = df_out["continuity_state"].astype(str)
    df_out["regime_id"] = df_out["regime_id"].astype(int)

    cur = conn.cursor()
    cur.executemany(f"""
        INSERT INTO continuity ({",".join(insert_cols)})
        VALUES ({",".join(["?"] * len(insert_cols))})
        ON CONFLICT(draw_date) DO UPDATE SET
          sum=excluded.sum,
          range=excluded.range,
          gap_mean=excluded.gap_mean,
          gap_std=excluded.gap_std,
          gap_max=excluded.gap_max,
          consec_pairs=excluded.consec_pairs,
          odd=excluded.odd,
          even=excluded.even,
          low=excluded.low,
          high=excluded.high,
          overlap_prev=excluded.overlap_prev,
          delta_mean_abs=excluded.delta_mean_abs,
          delta_max_abs=excluded.delta_max_abs,
          delta_neg=excluded.delta_neg,
          delta_zero=excluded.delta_zero,
          delta_pos=excluded.delta_pos,
          continuity_score=excluded.continuity_score,
          continuity_z=excluded.continuity_z,
          regime_break=excluded.regime_break,
          continuity_state=excluded.continuity_state,
          regime_id=excluded.regime_id
    """, df_out.values.tolist())
    conn.commit()

def write_transition_log(conn: sqlite3.Connection, log_df: pd.DataFrame) -> None:
    if log_df.empty:
        return
    cur = conn.cursor()
    # normalize dates to string
    x = log_df.copy()
    x["prev_date"] = pd.to_datetime(x["prev_date"]).dt.strftime("%Y-%m-%d")
    x["draw_date"] = pd.to_datetime(x["draw_date"]).dt.strftime("%Y-%m-%d")

    cur.executemany("""
        INSERT INTO state_transitions
        (prev_date, date, regime_id, prev_state, new_state, transition, continuity_score, continuity_z)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, x[["prev_date","draw_date","regime_id","prev_state","new_state","transition","continuity_score","continuity_z"]].values.tolist())
    conn.commit()

def write_break_explanations(conn: sqlite3.Connection, expl_df: pd.DataFrame) -> None:
    if expl_df.empty:
        return
    cur = conn.cursor()
    x = expl_df.copy()
    x["break_date"] = pd.to_datetime(x["break_date"]).dt.strftime("%Y-%m-%d")

    cur.executemany("""
        INSERT INTO break_explanations
        (break_date, regime_id, metric, before_mean, after_mean, delta, pct_change, continuity_z, state_at_break)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, x[["break_date","regime_id","metric","before_mean","after_mean","delta","pct_change","continuity_z","state_at_break"]].values.tolist())
    conn.commit()

# ----------------------------
# Continuity computations (7-number equal-member)
# ----------------------------

def _sorted_state(nums7: List[int]) -> List[int]:
    s = sorted(int(v) for v in nums7)
    if len(s) != 7:
        raise ValueError("Expected exactly 7 values (6 main + bonus).")
    for v in s:
        if v < 1 or v > 59:
            raise ValueError(f"Out of range 1..59: {v}")
    return s

def compute_continuity_features(df: pd.DataFrame, cols7: List[str]) -> pd.DataFrame:
    states = []
    for _, row in df.iterrows():
        nums7 = [int(row[c]) for c in cols7]
        states.append(_sorted_state(nums7))

    feats = []
    for i, now in enumerate(states):
        gaps = np.diff(now)
        base = {
            "sum": int(np.sum(now)),
            "range": int(now[-1] - now[0]),
            "gap_mean": float(np.mean(gaps)),
            "gap_std": float(np.std(gaps)),
            "gap_max": int(np.max(gaps)),
            "consec_pairs": int(np.sum(gaps == 1)),
        }
        arr = np.array(now, dtype=int)
        base.update({
            "odd": int(np.sum(arr % 2 == 1)),
            "even": int(np.sum(arr % 2 == 0)),
            "low": int(np.sum(arr <= 29)),
            "high": int(np.sum(arr >= 30)),
        })

        if i == 0:
            base.update({
                "overlap_prev": np.nan,
                "delta_mean_abs": np.nan,
                "delta_max_abs": np.nan,
                "delta_neg": np.nan,
                "delta_zero": np.nan,
                "delta_pos": np.nan,
            })
        else:
            prev = states[i - 1]
            base["overlap_prev"] = float(len(set(now).intersection(set(prev))))
            d = np.array(now, dtype=int) - np.array(prev, dtype=int)
            base.update({
                "delta_mean_abs": float(np.mean(np.abs(d))),
                "delta_max_abs": float(np.max(np.abs(d))),
                "delta_neg": float(np.sum(d < 0)),
                "delta_zero": float(np.sum(d == 0)),
                "delta_pos": float(np.sum(d > 0)),
            })

        feats.append(base)

    return pd.DataFrame(feats, index=df.index)

def compute_continuity_score(feat_df: pd.DataFrame, window: int = 30, z_threshold: float = 3.0) -> pd.DataFrame:
    use_cols = [
        "overlap_prev","sum","range","gap_std","gap_max","consec_pairs",
        "delta_mean_abs","delta_max_abs","delta_neg","delta_zero","delta_pos",
        "odd","low"
    ]
    use_cols = [c for c in use_cols if c in feat_df.columns]
    X = feat_df[use_cols].astype(float)

    mean_recent = X.rolling(window).mean()
    mean_prior = X.shift(window).rolling(window).mean()
    std_recent = X.rolling(window).std().replace(0, np.nan)

    diff = (mean_recent - mean_prior) / std_recent
    score = np.sqrt(np.nansum(diff ** 2, axis=1))
    score_s = pd.Series(score, index=feat_df.index)

    mu = score_s.rolling(window).mean()
    sd = score_s.rolling(window).std().replace(0, np.nan)
    z = (score_s - mu) / sd

    out = pd.DataFrame({
        "continuity_score": score_s,
        "continuity_z": z,
        "regime_break": (z >= z_threshold)
    }, index=feat_df.index)
    return out

# ----------------------------
# Continuity state machine + regime labeling + logging
# ----------------------------

STATE_STABLE = "STABLE"
STATE_DRIFT = "DRIFT"
STATE_BREAK = "BREAK"
STATE_SHOCK = "SHOCK"
STATE_RECOVERY = "RECOVERY"

def classify_level(z: float, z_low=1.5, z_break=3.0, z_shock=6.0) -> str:
    if pd.isna(z):
        return "UNKNOWN"
    if z >= z_shock:
        return STATE_SHOCK
    if z >= z_break:
        return STATE_BREAK
    if z >= z_low:
        return STATE_DRIFT
    return STATE_STABLE

def assign_fsm_state(df_out: pd.DataFrame, z_low=1.5, z_break=3.0, z_shock=6.0, r_recovery=3) -> pd.DataFrame:
    """
    Adds:
      - regime_level  (STABLE/DRIFT/BREAK/SHOCK)
      - continuity_state (STABLE/DRIFT/BREAK/SHOCK/RECOVERY)
    """
    df = df_out.copy()
    df["regime_level"] = df["continuity_z"].apply(lambda z: classify_level(z, z_low, z_break, z_shock))

    state = STATE_STABLE
    recovery_ok = 0
    states = []

    for lvl in df["regime_level"].tolist():
        if lvl in (STATE_BREAK, STATE_SHOCK):
            state = lvl
            recovery_ok = 0
            states.append(state)
            continue

        if state in (STATE_BREAK, STATE_SHOCK):
            state = STATE_RECOVERY
            recovery_ok = 0

        if state == STATE_RECOVERY:
            if lvl == STATE_STABLE:
                recovery_ok += 1
                if recovery_ok >= r_recovery:
                    state = STATE_STABLE
            else:
                recovery_ok = 0

        elif state == STATE_STABLE:
            if lvl == STATE_DRIFT:
                state = STATE_DRIFT

        elif state == STATE_DRIFT:
            if lvl == STATE_STABLE:
                state = STATE_STABLE

        states.append(state)

    df["continuity_state"] = states
    return df

def add_regime_id(df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Regime id increments on each break event (first TRUE of a run),
    using regime_break OR SHOCK.
    """
    df = df_out.copy()
    shock = df.get("regime_level", pd.Series([""] * len(df))).eq(STATE_SHOCK)
    br = df["regime_break"].fillna(False).astype(bool) | shock

    break_event = br & (~br.shift(1).fillna(False))
    df["break_event"] = break_event

    # Regime 1 before first break event
    df["regime_id"] = break_event.cumsum() + 1
    return df

def build_transition_log(df_out: pd.DataFrame) -> pd.DataFrame:
    """
    Logs only when continuity_state changes.
    Includes dates and key metrics.
    """
    cols = ["draw_date", "continuity_state", "continuity_z", "continuity_score", "regime_id"]
    x = df_out[cols].copy()

    x["prev_state"] = x["continuity_state"].shift(1)
    x["prev_date"] = x["draw_date"].shift(1)
    x["state_changed"] = (x["continuity_state"] != x["prev_state"]) & x["prev_state"].notna()

    log = x[x["state_changed"]].copy()
    log.rename(columns={"continuity_state": "new_state"}, inplace=True)
    log["transition"] = log["prev_state"] + " → " + log["new_state"]

    # Keep and order columns
    log = log[[
        "prev_date", "draw_date", "regime_id",
        "prev_state", "new_state", "transition",
        "continuity_score", "continuity_z"
    ]].reset_index(drop=True)
    return log

def build_break_explanations(df_out: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    For each break_event, compare mean metrics in the window BEFORE vs AFTER the break.
    Produces one row per metric per break date (tidy table).
    """
    metrics = [
        "sum", "range", "gap_std", "gap_max", "consec_pairs",
        "overlap_prev", "delta_mean_abs", "delta_max_abs",
        "delta_pos", "delta_neg", "odd", "low"
    ]
    metrics = [m for m in metrics if m in df_out.columns]

    rows = []
    idxs = df_out.index[df_out["break_event"] == True].tolist()

    for i in idxs:
        # define windows
        start_before = i - window
        end_before = i - 1
        start_after = i
        end_after = i + window - 1

        if start_before < 0 or end_after >= len(df_out):
            # skip edges (not enough context)
            continue

        before = df_out.loc[start_before:end_before, metrics].astype(float)
        after = df_out.loc[start_after:end_after, metrics].astype(float)

        before_mean = before.mean(numeric_only=True)
        after_mean = after.mean(numeric_only=True)
        delta = after_mean - before_mean

        for m in metrics:
            b = float(before_mean[m])
            a = float(after_mean[m])
            d = float(delta[m])
            pct = (d / b * 100.0) if (b != 0 and not np.isnan(b)) else np.nan

            rows.append({
                "break_date": df_out.loc[i, "draw_date"],
                "regime_id": int(df_out.loc[i, "regime_id"]),
                "metric": m,
                "before_mean": b,
                "after_mean": a,
                "delta": d,
                "pct_change": pct,
                "continuity_z": float(df_out.loc[i, "continuity_z"]) if not pd.isna(df_out.loc[i, "continuity_z"]) else np.nan,
                "state_at_break": df_out.loc[i, "continuity_state"]
            })

    return pd.DataFrame(rows)

# ----------------------------
# Data preparation helpers
# ----------------------------

def ensure_num_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts either:
      - draw_date + num1..num6 + bonus
      - draw_date + winning_numbers + bonus   (winning_numbers is "02 18 24 27 29 39")
    Returns df with draw_date + num1..num6 + bonus as ints, sorted by draw_date asc.
    """
    df = df.copy()

    if "winning_numbers" in df.columns and not all(c in df.columns for c in ["num1","num2","num3","num4","num5","num6"]):
        parts = df["winning_numbers"].astype(str).str.strip().str.split(r"\s+", expand=True)
        if parts.shape[1] < 6:
            raise ValueError("winning_numbers does not split into 6 values.")
        parts = parts.iloc[:, :6].astype(int)
        parts.columns = ["num1","num2","num3","num4","num5","num6"]
        df = pd.concat([df.drop(columns=[]), parts], axis=1)

    required = ["draw_date","num1","num2","num3","num4","num5","num6","bonus"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["draw_date"] = pd.to_datetime(df["draw_date"])
    for c in ["num1","num2","num3","num4","num5","num6","bonus"]:
        df[c] = df[c].astype(int)

    df = df.sort_values("draw_date", ascending=True).reset_index(drop=True)
    return df

# ----------------------------
# Replit page
# ----------------------------

def render_continuity_page(df_raw: pd.DataFrame) -> None:
    st.header("Continuity Dashboard (Lotto 6/59 + Bonus)")

    with st.expander("Settings", expanded=True):
        window = st.slider("Continuity window (draws)", min_value=10, max_value=120, value=30, step=5)
        z_threshold = st.slider("Regime break threshold (z-score)", min_value=1.5, max_value=6.0, value=3.0, step=0.5)

        persist = st.checkbox("Persist draws + continuity to SQLite", value=True)
        db_path = st.text_input("SQLite DB path", value=DB_PATH)

    df = ensure_num_columns(df_raw)

    cols7 = ["num1","num2","num3","num4","num5","num6","bonus"]
    feat_df = compute_continuity_features(df, cols7)
    score_df = compute_continuity_score(feat_df, window=window, z_threshold=z_threshold)

    df_out = pd.concat([df, feat_df, score_df], axis=1)

    # ---- NEW: FSM + regime ids + logs + explanations ----
    z_low = 1.5
    z_break = z_threshold          # keep your slider as the break threshold
    z_shock = 6.0                  # fixed "shock" definition (can be a slider later)
    r_recovery = 3

    df_out = assign_fsm_state(df_out, z_low=z_low, z_break=z_break, z_shock=z_shock, r_recovery=r_recovery)
    df_out = add_regime_id(df_out)

    transition_log = build_transition_log(df_out)
    break_expl = build_break_explanations(df_out, window=window)

    st.subheader("Current continuity mode")
    st.write(f"Current state: **{df_out.iloc[-1]['continuity_state']}**")
    st.write(f"Current regime_id: **{int(df_out.iloc[-1]['regime_id'])}**")

    # Persist
    if persist:
        conn = sqlite3.connect(db_path)
        init_db(conn)
        upsert_draws(conn, df_out[["draw_date"] + cols7])
        upsert_continuity(conn, df_out[["draw_date"] + feat_df.columns.tolist() + score_df.columns.tolist() + ["continuity_state","regime_id"]])

        write_transition_log(conn, transition_log)
        write_break_explanations(conn, break_expl)

        conn.close()
        st.caption(f"Persisted draws, continuity, transitions, and break explanations to SQLite: {db_path}")

    # Plot continuity score
    st.subheader("Continuity score over time")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_out["draw_date"], df_out["continuity_score"], linewidth=1)
    ax.set_xlabel("Draw Date")
    ax.set_ylabel("Continuity Score")
    ax.set_title("Continuity Score (higher = more likely regime shift)")
    st.pyplot(fig)
    
    # Download Chart 1
    import io
    img1 = io.BytesIO()
    fig.savefig(img1, format='png')
    st.download_button(
        label="Download Continuity Score Chart",
        data=img1.getvalue(),
        file_name="continuity_score.png",
        mime="image/png"
    )

    # Plot z-score with break markers
    st.subheader("Regime breaks (z-score)")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df_out["draw_date"], df_out["continuity_z"], linewidth=1)
    breaks = df_out[df_out["regime_break"] == True]
    if not breaks.empty:
        ax2.scatter(breaks["draw_date"], breaks["continuity_z"], s=25)
    ax2.axhline(y=z_threshold, linestyle="--", linewidth=1)
    ax2.set_xlabel("Draw Date")
    ax2.set_ylabel("Continuity Z-score")
    ax2.set_title("Continuity Z-score with break threshold")
    st.pyplot(fig2)

    # Download Chart 2
    img2 = io.BytesIO()
    fig2.savefig(img2, format='png')
    st.download_button(
        label="Download Regime Breaks Chart",
        data=img2.getvalue(),
        file_name="regime_breaks.png",
        mime="image/png"
    )

    # Display transition log
    st.subheader("State transition log")
    st.dataframe(transition_log.tail(30), use_container_width=True)

    # Display break explanation table
    st.subheader("Break explanations (what changed)")
    if break_expl.empty:
        st.info("No break explanations available (need enough data before/after breaks for the chosen window).")
    else:
        # Let user choose a break date
        available_dates = break_expl["break_date"].drop_duplicates().sort_values()
        chosen = st.selectbox("Select break date", available_dates.astype(str).tolist(), index=len(available_dates)-1)
        view = break_expl[break_expl["break_date"].astype(str) == str(chosen)].copy()

        # show largest changes first
        view = view.sort_values("pct_change", ascending=False, key=lambda s: s.abs() if s.notna().any() else s)

        st.dataframe(view, use_container_width=True)

    # Break table
    st.subheader("Detected breakpoints")
    show_cols = ["draw_date","continuity_score","continuity_z","overlap_prev","sum","range","gap_std","delta_mean_abs","delta_max_abs"]
    show_cols = [c for c in show_cols if c in df_out.columns]
    st.dataframe(
        df_out[df_out["regime_break"] == True][show_cols].sort_values("draw_date", ascending=False),
        use_container_width=True
    )

    # Recent features
    st.subheader("Most recent draws (with continuity features)")
    tail_cols = ["draw_date"] + cols7 + ["overlap_prev","continuity_score","continuity_z","regime_break"]
    tail_cols = [c for c in tail_cols if c in df_out.columns]
    st.dataframe(df_out[tail_cols].tail(20).sort_values("draw_date", ascending=False), use_container_width=True)

    # Download CSV
    st.subheader("Download Analysis Results")
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download full analysis as CSV",
        data=csv,
        file_name=f"lotto_analysis_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )
