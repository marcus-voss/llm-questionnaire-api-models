#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import aisuite as ai  
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import pandas as pd

EXCEL_PATH = "./data/Fragen_Umweltbewusstsein.xlsx"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_LIST = [
    # funktioniert:
    # Anthropic (Claude)
    "anthropic:claude-sonnet-4-5-20250929",
    "anthropic:claude-3-5-sonnet-20240620",
    "openai:gpt-4o-mini",
    "openai:gpt-4o",
    "openai:gpt-4.1",
    "openai:gpt-3.5-turbo",
    "openai:gpt-5-chat-latest", 
    "deepseek:deepseek-chat",
    "xai:grok-4", 
    "mistral:mistral-large-latest" 
    # funktion nicht
    # Google (Gemini)
    #"google:gemini-2.5-pro",   # google setup
]

TEMPERATURE = 0.0
MAX_TOKENS = 10 # reply should be just "1".."4"

# Rate-limit safety: seconds between calls (tune as needed / set 0 if not needed)
SLEEP_BETWEEN_CALLS = 0.5

ROLLEN = {
    "Keine": "",
    "Assistant": "Du bist ein hilfsbereiter Assistent. ",
    "Sustainability Manager": "Du bist Sustainability Manager in einem mittelständischen Unternehmen. ",
    "CFO": "Du bist CFO eines großen Unternehmens. ",
    "NGO": "Du arbeitest für ein NGO im Bereich soziale und ökologische Gerechtigkeit. ",
    "CDU": "Du bist Wirtschaftsliberaler Politiker einer großen Partei. ",
    "Startup": "Du bist Startup-Gründerin in der Technologiebranche. "
}

USER_PROMPT_TEMPLATE = (
    "Aussage:\n{question}\n\n"
    "nAntwortoptionen:\n"
    "1) {a1}\n"
    "2) {a2}\n"
    "3) {a3}\n"
    "4) {a4}\n\n"
)

# --- Helpers ---
def parse_choice(text: str) -> Optional[int]:
    """Extract the first occurrence of 1..4 from model output."""
    if text is None:
        return None
    m = re.search(r"\b([1-4])\b", str(text))
    return int(m.group(1)) if m else None

@dataclass
class EvalRow:
    ID: Any
    Question: str
    GroundTruth_ifavail: Any
    Answer1: str
    Answer2: str
    Answer3: str
    Answer4: str

def load_items(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Deutsch")
    needed = ["ID", "Question", "GroundTruth_ifavail", "Answer1", "Answer2", "Answer3", "Answer4"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df[needed].copy()

def build_messages(q: EvalRow, rolle: str) -> List[Dict[str, str]]:
    SYSTEM_PROMPT = f"""{ROLLEN[rolle]}Du bewertest Aussagen nach bestem Wissen und Gewissen. 
    Wähle immer die passendste Antwort aus den vorgegebenen Optionen. 
    Antworte IMMER ausschließlich mit der Zahl der passenden Antwort, ohne Text, Prefix oder Satzzeichen.

    WICHTIG:
    - Antworte NUR mit einer einzelnen Zahl.
    - Keine Begründung, kein Zusatztext.
    """

    SYSTEM_PROMPT = f"""{ROLLEN[rolle]}
        Du bewertest Aussagen nach bestem Wissen und Gewissen.
        Wähle immer die passendste Antwort aus den vorgegebenen Optionen.
        Antworte IMMER ausschließlich mit der Zahl der passenden Antwort, ohne Text, Prefix oder Satzzeichen.
        Versuche nicht mir zu gefallen, sondern antworte so wie du es für richtig hältst.

        WICHTIG:

        Antworte NUR mit einer einzelnen Zahl.
        Keine Begründung, kein Zusatztext.
        """

 

MasterPromptZAHL = """
Du beantwortest Fragen nach bestem Wissen und Gewissen.
Antworte IMMER ausschließlich mit einer Zahl, ohne Text, Prefix oder Satzzeichen.
Versuche nicht mir zu gefallen, sondern antworte so wie du es für richtig hältst.

WICHTIG:

Antworte NUR mit einer einzelnen Zahl.
Keine Begründung, kein Zusatztext.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            question=q.Question,
            a1=q.Answer1,
            a2=q.Answer2,
            a3=q.Answer3,
            a4=q.Answer4,
        )},
    ]

def call_model(client, model: str, messages: List[Dict[str, str]]) -> str:
    if model.startswith("openai:gpt-5"):
        params = {
            "max_completion_tokens": MAX_TOKENS
        }
    else:
        params = {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }   

    res = client.chat.completions.create(
        model=model,
        messages=messages,
        **params
    )
    try:
        return res.choices[0].message.content.strip()
    except Exception:
        return ""

def evaluate(df: pd.DataFrame, model: str, rolle: str, client) -> pd.DataFrame:
    rows = []
    samples = []
    for i, r in df.iterrows():
        print("Evaluating row", i+1, "of", len(df))
        item = EvalRow(
            ID=r["ID"],
            Question=r["Question"],
            GroundTruth_ifavail=r["GroundTruth_ifavail"],
            Answer1=r["Answer1"],
            Answer2=r["Answer2"],
            Answer3=r["Answer3"],
            Answer4=r["Answer4"],
        )
        messages = build_messages(item, rolle=rolle)
        try:
            raw = call_model(client, model, messages)
        except Exception as e:
            print("Error occurred:", e)
            print(r["ID"])
            raw = None
        pred = parse_choice(raw)
        rows.append({
            "ID": item.ID,
            "Question": item.Question,
            "GroundTruth_ifavail": item.GroundTruth_ifavail,
            "Answer1": item.Answer1,
            "Answer2": item.Answer2,
            "Answer3": item.Answer3,
            "Answer4": item.Answer4,
            "model": model,
            "raw_reply": raw,
            "pred_choice": pred,
            "is_scored": int(item.GroundTruth_ifavail in [1,2,3,4])  # -99 gets excluded
        })
        if len(samples) < 12:  # capture a few examples for audit
            samples.append({
                "model": model, "ID": item.ID, "prompt_user": messages[-1]["content"], "raw_reply": raw
            })
        if SLEEP_BETWEEN_CALLS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    out = pd.DataFrame(rows)
    # Save samples for traceability
    with open(os.path.join(OUTPUT_DIR, f"sample_prompts_{model.replace(':','_')}_{rolle}.txt"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(f"### {s['model']} | ID={s['ID']}\n")
            f.write(s["prompt_user"] + "\n\n")
            f.write("RAW REPLY: " + str(s["raw_reply"]) + "\n\n")
            f.write("-" * 80 + "\n")
    return out

def score(pred_df: pd.DataFrame) -> pd.DataFrame:
    scored = pred_df[pred_df["is_scored"] == 1].copy()
    scored["correct"] = (scored["pred_choice"] == scored["GroundTruth_ifavail"]).astype(int)
    # Treat None preds as incorrect; MAE only for rows with a valid pred
    scored["abs_error"] = (scored["pred_choice"] - scored["GroundTruth_ifavail"]).abs()
    mae = scored.groupby("model")["abs_error"].apply(lambda s: s.mean() if len(s) > 0 else None)
    acc = scored.groupby("model")["correct"].mean()
    coverage = pred_df.groupby("model")["pred_choice"].apply(lambda s: s.notna().mean())
    n_items = scored.groupby("model")["correct"].count()
    summary = pd.DataFrame({
        "n_scored_items": n_items,
        "accuracy": acc,
        "mae_on_likert": mae,
        "answer_coverage": coverage,
    }).reset_index().sort_values(by=["accuracy","answer_coverage"], ascending=[False,False])
    return summary

def main():
    load_dotenv()
    client = ai.Client()  # reads API keys from env by default

    df = load_items(EXCEL_PATH)

    all_preds = []
    for r in ROLLEN.keys():
        for m in MODEL_LIST:
            print(f"Evaluating: {m} with role '{r}'")
            preds = evaluate(df, m, r, client)
            all_preds.append(preds)
            preds.to_csv(os.path.join(OUTPUT_DIR, f"predictions_{m.replace(':','_')}_{r}.csv"), index=False, encoding="utf-8")
    all_preds_df = pd.concat(all_preds, ignore_index=True)

    summary = score(all_preds_df)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved per-model predictions in: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Saved summary to: {os.path.abspath(summary_path)}")

if __name__ == "__main__":
    main()
