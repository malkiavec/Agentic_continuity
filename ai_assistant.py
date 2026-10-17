import json
import os
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st


API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CHAT_KEY = "continuity_ai_assistant_messages"


def _analysis_context(
    df_out: pd.DataFrame,
    transition_log: pd.DataFrame,
    break_expl: pd.DataFrame,
) -> Dict[str, Any]:
    recent_cols = [
        col
        for col in (
            "draw_date",
            "num1",
            "num2",
            "num3",
            "num4",
            "num5",
            "num6",
            "bonus",
            "continuity_score",
            "continuity_z",
            "continuity_state",
            "regime_id",
        )
        if col in df_out.columns
    ]
    recent = df_out[recent_cols].tail(10).copy()
    if "draw_date" in recent.columns:
        recent["draw_date"] = pd.to_datetime(recent["draw_date"]).dt.strftime("%Y-%m-%d")
    recent = recent.astype(object).where(pd.notna(recent), None)

    latest = df_out.iloc[-1]
    break_rows = df_out[df_out["regime_break"].fillna(False).astype(bool)]
    transitions = transition_log.tail(8).copy()
    if "draw_date" in transitions.columns:
        transitions["draw_date"] = pd.to_datetime(transitions["draw_date"]).dt.strftime("%Y-%m-%d")
    transitions = transitions.astype(object).where(pd.notna(transitions), None)

    latest_break_details = []
    if not break_expl.empty:
        most_recent_break = break_expl["break_date"].max()
        recent_break_metrics = (
            break_expl[break_expl["break_date"] == most_recent_break]
            .sort_values("pct_change", key=lambda values: values.abs(), ascending=False)
            .head(8)
        )
        latest_break_details = (
            recent_break_metrics.astype(object)
            .where(pd.notna(recent_break_metrics), None)
            .to_dict(orient="records")
        )

    return {
        "draw_count": int(len(df_out)),
        "date_range": {
            "first": pd.to_datetime(df_out["draw_date"].min()).strftime("%Y-%m-%d"),
            "latest": pd.to_datetime(df_out["draw_date"].max()).strftime("%Y-%m-%d"),
        },
        "latest_state": str(latest.get("continuity_state", "UNKNOWN")),
        "latest_regime_id": int(latest.get("regime_id", 0)),
        "latest_continuity_score": (
            None if pd.isna(latest.get("continuity_score")) else float(latest["continuity_score"])
        ),
        "latest_continuity_z": (
            None if pd.isna(latest.get("continuity_z")) else float(latest["continuity_z"])
        ),
        "breakpoint_count": int(len(break_rows)),
        "recent_draws_and_metrics": recent.to_dict(orient="records"),
        "recent_state_transitions": transitions.to_dict(orient="records"),
        "most_recent_break_metric_changes": latest_break_details,
    }


def _request_assistant_reply(messages: list) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("The OpenAI API key is not configured.")

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 700,
        },
        timeout=(10, 60),
    )

    if response.status_code == 401:
        raise RuntimeError("The configured AI key was rejected. Check the OPENAI_API_KEY secret.")
    if response.status_code == 429:
        raise RuntimeError("The AI provider is rate-limiting requests or has reached its usage limit.")
    if response.status_code >= 500:
        raise RuntimeError("The AI provider is temporarily unavailable. Please try again shortly.")
    response.raise_for_status()

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("The AI provider returned an unexpected response.") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The AI provider returned an empty response.")
    return content.strip()


def render_ai_assistant(
    df_out: pd.DataFrame,
    transition_log: pd.DataFrame,
    break_expl: pd.DataFrame,
) -> None:
    st.subheader("AI Assistant")
    st.caption(
        "Ask questions about the continuity analysis. Lottery draws are random; "
        "historical patterns cannot predict future results."
    )

    if not os.getenv("OPENAI_API_KEY"):
        st.info(
            "To enable chat, add an OpenAI API key as the Replit secret "
            "`OPENAI_API_KEY`. The key is used only by the server and is never shown here."
        )
        return

    messages = st.session_state.setdefault(CHAT_KEY, [])
    controls, _ = st.columns([1, 5])
    with controls:
        if st.button("Clear chat", key="clear_continuity_ai_chat"):
            st.session_state[CHAT_KEY] = []
            messages = st.session_state[CHAT_KEY]

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about the continuity results…", key="continuity_ai_chat_input")
    if not prompt:
        return

    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    system_message = {
        "role": "system",
        "content": (
            "You are the AI assistant for a lottery continuity-analysis dashboard. "
            "Explain the supplied analysis clearly, distinguish observations from conclusions, "
            "and state uncertainty. The analysis detects statistical changes in historical draws; "
            "it does not establish that future lottery numbers can be predicted. Never claim a "
            "number, pattern, or strategy improves the odds, and do not provide guaranteed picks. "
            "If asked to predict winning numbers, explain that draws are random and offer to "
            "interpret the historical metrics instead. Use only the supplied analysis context "
            "for claims about this dashboard.\n\n"
            f"Current analysis context:\n{json.dumps(_analysis_context(df_out, transition_log, break_expl))}"
        ),
    }
    request_messages = [system_message] + messages[-16:]

    with st.chat_message("assistant"):
        with st.spinner("Analyzing the dashboard context…"):
            try:
                reply = _request_assistant_reply(request_messages)
            except requests.Timeout:
                reply = "The AI request timed out. Please try again."
            except requests.RequestException:
                reply = "Could not reach the AI provider. Check the connection and try again."
            except RuntimeError as exc:
                reply = str(exc)
            except Exception:
                reply = "The assistant encountered an unexpected error. Please try again."
        st.markdown(reply)
    messages.append({"role": "assistant", "content": reply})
