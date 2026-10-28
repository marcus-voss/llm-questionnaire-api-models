#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import aisuite as ai  
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd

EXCEL_PATH = "./data/Fragen_Umweltbewusstsein.xlsx"
SHEET_NAME = "Deutsch_v3"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models are addressed through aisuite as "<provider>:<model>".
# The five models below reproduce the proprietary-model results reported in the
# paper (and the shipped example_output/). Each requires the matching API key in
# .env (see .env.example).
MODEL_LIST = [
    "anthropic:claude-sonnet-4-6",
    "openai:gpt-5.2-2025-12-11",        # reasoning model
    "deepseek:deepseek-chat",           # DeepSeek-V3.2
    "xai:grok-4-1-fast-non-reasoning",
    "mistral:mistral-large-2512",

    # Additional models we also evaluated – uncomment as needed:
    # "anthropic:claude-opus-4-6",
    # "anthropic:claude-haiku-4-5-20251001",
    # "openai:gpt-4.1-2025-04-14",            # non-reasoning
    # "mistral:mistral-medium-2508",
    # "mistral:mistral-small-2506",
    # "google:gemini-2.5-flash",              # needs a Google Cloud service account
    # "google:gemini-2.5-pro",
]

TEMPERATURE = 0.0
MAX_TOKENS = 200  # more tokens allow model to reason briefly; we parse the last number
# Bootstrap: number of repeated calls per question (median is used as final answer, like src/)
N_BOOTSTRAP = 3
# Rate-limit safety: seconds between calls (tune as needed / set 0 if not needed)
SLEEP_BETWEEN_CALLS = 1.0
# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # exponential backoff: 2, 4, 8 seconds
# Parallelization: runs different providers in parallel, but sequential within each provider
# Set to 1 to disable parallelization completely
MAX_PARALLEL_PROVIDERS = 4

# Role / perspective prompts prepended to every question. These 11 roles are the
# ones used for the paper (first-person "(ich)" professional roles plus the five
# UBA environmental-awareness milieus); see example_output/ for a full run.
ROLLEN = {
    "Nachhaltigkeitsmanager (ich)": "Ich bin ein/e Nachhaltigkeitsmanager/in in einem mittelständischen Unternehmen. ",
    "CFO (ich)": "Ich bin CFO in einem großen Unternehmen. ",
    "NGO (ich)": "Ich bin für eine NGO im Bereich “soziale und ökologische Gerechtigkeit” tätig. ",
    "Wirtschaftsliberal (ich)": "Ich bin wirtschaftsliberale/r Politiker/in in einer großen Partei. ",
    "Startup (ich)": "Ich bin Startup-Gründer/in in der Technologiebranche. ",
    "Wissenschaft (ich)": "Ich bin ein/e Wissenschaftler/in an einem deutschen Forschungsinstitut. ",
    "UBA Engagierte": "Ich bin sehr umweltbewusst und engagiert. Ich fordere eine konsequentere Umwelt- und Klimapolitik. Aus meiner Sicht muss der Schutz von Umwelt und Klima Priorität vor dem Wirtschaftswachstum haben. ",
    "UBA Individuell-Nachhaltige": "Ich lebe umweltbewusst. Ich versuche, mich im Rahmen meiner persönlichen Möglichkeiten nachhaltig zu verhalten. Von der Politik erwarte ich eine stärkere Ausrichtung auf Umwelt- und Klimaschutz. ",
    "UBA Ambivalente": "Ich bin besorgt angesichts der Klima- und Umweltkrise. Ich sehe für mich selbst aber wenig Möglichkeiten, etwas zum Klima- und Umweltschutz beizutragen.  Ich mache mir Sorgen, dass  klimapolitische Maßnahmen negative soziale Folgen haben. ",
    "UBA Populistisch-Ablehnende": "Ich halte die ökologische Krise für nicht so dramatisch. Ich finde, die Politik sollte sich nicht so viel um den Schutz von Umwelt und Klima kümmern, sondern mehr um  die Bedürfnisse der einfachen Leute. ",
    "UBA Neoliberal-Ablehnende": "Ich finde die Angst, die viele vor dem Klimawandel haben, übertrieben. Die Politik soll es unterlassen, zum Schutz von Umwelt und Klima  individuelle und unternehmerische Freiheiten einzuschränken, da das negative Folgen für die Wirtschaft hat. ",
}

def parse_choice(text: str) -> Optional[float]:
    """Extract the LAST occurrence of a number (int or float) from model output.
    Uses word boundaries (matching src/llm.py) to avoid catching numbers embedded
    in longer strings (e.g. model version numbers like '2407').
    Supports both '.' and ',' as decimal separators (German locale)."""
    if text is None:
        return None
    matches = re.findall(r"\b\d+(?:[.,]\d+)?\b", str(text))
    if matches:
        return float(matches[-1].replace(",", "."))
    return None


def compute_median(values: list) -> Optional[float]:
    """Compute median over bootstrap samples, ignoring negative sentinel values (-99).
    Returns None if no valid values remain (mirrors src/utils/helper.py)."""
    filtered = [v for v in values if v is not None and v >= 0]
    if not filtered:
        return None
    return float(np.median(filtered))

@dataclass
class EvalRow:
    ID: Any
    Question: str
    GroundTruth_ifavail: Any
    Answer1: str
    Answer2: str
    Answer3: str
    Answer4: str

def load_items(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (eval_df, train_df). train_df contains rows with IDs starting with 'Example' (used for few-shot).
    eval_df contains ALL rows including Examples."""
    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    needed = ["ID", "Question", "GroundTruth_ifavail", "Answer1", "Answer2", "Answer3", "Answer4"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df = df[needed].copy().reset_index(drop=True)  # preserve original Excel row order
    df.insert(0, "row_index", df.index)             # original 0-based row position
    train_df = df[df["ID"].astype(str).str.startswith("Example")].reset_index(drop=True)
    eval_df  = df  # include all rows (Examples + Test + others), original order
    return eval_df, train_df


def build_fewshot_block(train_df: pd.DataFrame) -> str:
    """Build a few-shot examples block from training rows.
    Spacing and wording matches src/utils/helper.py exactly."""
    blocks = []
    for _, r in train_df.iterrows():
        if str(r["Answer1"]) != "ZAHL":
            formatted = (
                f"Aussage: {r['Question']}\nAntwortoptionen:\n"
                + "\n".join(f"{i+1}: {r[f'Answer{i+1}']}" for i in range(4))
            )
            block = f"{formatted}\n Richtige Antwort: {int(r['GroundTruth_ifavail'])}"
        else:
            block = f"{r['Question']}\n Richtige Antwort: {r['GroundTruth_ifavail']}"
        blocks.append(block)
    if not blocks:
        return ""
    # Exact spacing from src/utils/helper.py: " Hier einige Beispiele: \n", "\n \n".join, "\n \n Jetzt..."
    return (
        "\n Hier einige Beispiele: \n"
        + "\n \n".join(blocks)
        + "\n \n Jetzt bewerte die neue Aussage."
    )

def build_messages(q: EvalRow, rolle: str, fewshot_block: str = "") -> List[Dict[str, str]]:
    """Build a single-message prompt (no system/user split), matching src/ behaviour."""

    if q.Answer1 != "ZAHL":
        instructions = (
            f"{ROLLEN[rolle]}"
            "Du bewertest Aussagen nach den besten dir zur Verfügung stehenden Informationen.\n"
            "Wähle immer die passendste Antwort aus den vorgegebenen Optionen.\n"
            "Falls relevant: ich wohne in einem Zweipersonenhaushalt in einer deutschen Großstadt.\n"
            "Antworte immer ausschließlich mit der Zahl der passendsten Antwort, ohne Text, Prefix oder Satzzeichen.\n\n"
            "WICHTIG:\n"
            "- Antworte NUR mit einer einzelnen Zahl.\n"
            "- Keine Begründung, kein Zusatztext.\n"
        )
        question_block = (
            f"Aussage: {q.Question}\n"
            f"Antwortoptionen:\n"
            f"1: {q.Answer1}\n"
            f"2: {q.Answer2}\n"
            f"3: {q.Answer3}\n"
            f"4: {q.Answer4}\n"
        )
    else:
        instructions = (
            f"{ROLLEN[rolle]}"
            "Du beantwortest Fragen nach den besten dir zur Verfügung stehenden Informationen.\n"
            "Falls relevant: ich wohne in einem Zweipersonenhaushalt in einer deutschen Großstadt.\n"
            "Antworte immer ausschließlich mit einer Zahl, ohne Text, Prefix oder Satzzeichen.\n\n"
            "WICHTIG:\n"
            "- Antworte NUR mit einer einzelnen Zahl.\n"
            "- Keine Begründung, kein Zusatztext.\n"
        )
        question_block = q.Question

    parts = [instructions]
    # Few-shot examples are MC-only (answers 1-4) – don't inject them for ZAHL questions
    # as they would bias the model toward answering with 1–4 instead of a real number.
    if fewshot_block and q.Answer1 != "ZAHL":
        parts.append(fewshot_block)
    parts.append(question_block)

    user_content = "\n".join(parts)
    return [{"role": "user", "content": user_content}]

def call_model_with_retry(client, model: str, messages: List[Dict[str, str]]) -> tuple[str, str | None]:
    """Call model with retry logic. Returns (response, error_message)."""
    if model.startswith("openai:gpt-5"):
        params = {
            "max_completion_tokens": MAX_TOKENS
        }
    else:
        params = {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        }   

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                **params
            )
            content = res.choices[0].message.content
            return (content.strip() if content else "", None)
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY_BASE ** (attempt + 1)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} after {wait_time}s: {e}")
                time.sleep(wait_time)
    
    return (None, last_error)

# Lock for thread-safe printing
print_lock = Lock()

def process_single_item(args) -> tuple[dict, dict]:
    """Process a single evaluation item with N_BOOTSTRAP repeated calls.
    Returns (idx, row_dict, sample_dict).
    The CSV will contain individual bootstrap raw replies + parsed values,
    the median as pred_choice, and a flag if all bootstrap answers agreed."""
    idx, r, model, rolle, client, fewshot_block = args

    item = EvalRow(
        ID=r["ID"],
        Question=r["Question"],
        GroundTruth_ifavail=r["GroundTruth_ifavail"],
        Answer1=r["Answer1"],
        Answer2=r["Answer2"],
        Answer3=r["Answer3"],
        Answer4=r["Answer4"],
    )
    messages = build_messages(item, rolle=rolle, fewshot_block=fewshot_block)

    raw_replies = []
    parsed_values = []
    errors = []

    for _ in range(N_BOOTSTRAP):
        raw, error_msg = call_model_with_retry(client, model, messages)
        if error_msg:
            with print_lock:
                print(f"  Error for ID {r['ID']}: {error_msg}")
        raw_replies.append(raw)
        parsed_values.append(parse_choice(raw))
        errors.append(error_msg)
        if SLEEP_BETWEEN_CALLS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    pred_median = compute_median(parsed_values)
    valid_preds = [v for v in parsed_values if v is not None]
    bootstrap_consistent = len(set(valid_preds)) <= 1  # True if all non-None answers agree

    bootstrap_cols = {}
    for i, (rr, pv) in enumerate(zip(raw_replies, parsed_values), start=1):
        bootstrap_cols[f"raw_reply_{i}"] = rr
        bootstrap_cols[f"pred_choice_{i}"] = pv

    row = {
        "ID": item.ID,
        "row_index": r["row_index"],
        "Question": item.Question,
        "GroundTruth_ifavail": item.GroundTruth_ifavail,
        "Answer1": item.Answer1,
        "Answer2": item.Answer2,
        "Answer3": item.Answer3,
        "Answer4": item.Answer4,
        "model": model,
        "rolle": rolle,
        **bootstrap_cols,
        "pred_choice": pred_median,         # median over bootstrap runs
        "bootstrap_consistent": bootstrap_consistent,  # False = model was uncertain
        "is_scored": int(item.GroundTruth_ifavail in [1, 2, 3, 4]),
        "error": "; ".join(e for e in errors if e),
    }
    # Keep first raw reply in sample log for readability
    sample = {
        "model": model, "ID": item.ID,
        "prompt_user": messages[-1]["content"],
        "raw_replies": raw_replies,
    }

    return (idx, row, sample)

def evaluate(df: pd.DataFrame, model: str, rolle: str, client, fewshot_block: str = "") -> pd.DataFrame:
    """Evaluate all items sequentially (parallelization happens at provider level)."""
    rows = []
    samples = []
    total = len(df)
    
    for i, r in df.iterrows():
        idx, row, sample = process_single_item((i, r, model, rolle, client, fewshot_block))
        rows.append(row)
        samples.append(sample)
        with print_lock:
            print(f"  {model}: {i+1}/{total}", end="\r")
    
    print() 
    
    out = pd.DataFrame(rows)
    # Save samples for traceability
    with open(os.path.join(OUTPUT_DIR, f"sample_prompts_{model.replace(':','_')}_{rolle}.txt"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(f"### {s['model']} | ID={s['ID']}\n")
            f.write(s["prompt_user"] + "\n\n")
            for i, rr in enumerate(s["raw_replies"], start=1):
                f.write(f"RAW REPLY [{i}/{N_BOOTSTRAP}]: {rr}\n")
            f.write("\n" + "-" * 80 + "\n")
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
    }).reset_index().sort_values(by=["accuracy", "answer_coverage"], ascending=[False,False])
    return summary

def get_provider(model: str) -> str:
    """Extract provider name from model string (e.g., 'anthropic:claude-3' -> 'anthropic')."""
    return model.split(":")[0] if ":" in model else model

def evaluate_model_rolle(args) -> pd.DataFrame:
    """Evaluate a single model/rolle combination. Used for parallel execution."""
    model, rolle, df, client, fewshot_block = args
    with print_lock:
        print(f"Starting: {model} with role '{rolle}'")
    
    preds = evaluate(df, model, rolle, client, fewshot_block)
    
    output_path = os.path.join(OUTPUT_DIR, f"predictions_{model.replace(':','_')}_{rolle}.csv")
    preds.to_csv(output_path, index=False, encoding="utf-8")
    
    with print_lock:
        print(f"Completed: {model} with role '{rolle}'")
    
    return preds

def run_provider_tasks(provider_tasks: List[tuple], client) -> List[pd.DataFrame]:
    """Run all tasks for a single provider sequentially."""
    results = []
    for model, rolle, df, fewshot_block in provider_tasks:
        preds = evaluate_model_rolle((model, rolle, df, client, fewshot_block))
        results.append(preds)
    return results

def main():
    load_dotenv()
    client = ai.Client()

    eval_df, train_df = load_items(EXCEL_PATH)
    fewshot_block = build_fewshot_block(train_df)
    if fewshot_block:
        print(f"Few-shot: {len(train_df)} training example(s) loaded from Excel.")
    else:
        print("Few-shot: no 'Example' rows found in Excel – running zero-shot.")

    # Group tasks by provider
    tasks_by_provider: Dict[str, List[tuple]] = {}
    for rolle in ROLLEN.keys():
        for model in MODEL_LIST:
            provider = get_provider(model)
            if provider not in tasks_by_provider:
                tasks_by_provider[provider] = []
            tasks_by_provider[provider].append((model, rolle, eval_df, fewshot_block))
    
    print(f"\nTask distribution by provider:")
    for provider, tasks in tasks_by_provider.items():
        print(f"  {provider}: {len(tasks)} tasks")
    print()

    all_preds = []
    
    # Run providers in parallel, but tasks within each provider sequentially
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PROVIDERS) as executor:
        futures = {
            executor.submit(run_provider_tasks, tasks, client): provider
            for provider, tasks in tasks_by_provider.items()
        }
        
        for future in as_completed(futures):
            provider = futures[future]
            try:
                provider_results = future.result()
                all_preds.extend(provider_results)
                print(f"\n✓ Provider '{provider}' completed ({len(provider_results)} tasks)")
            except Exception as e:
                print(f"\n✗ Provider '{provider}' failed: {e}")
    
    if all_preds:
        all_preds_df = pd.concat(all_preds, ignore_index=True)
        summary = score(all_preds_df)
        summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
        summary.to_csv(summary_path, index=False)
        print("\n=== Summary ===")
        print(summary.to_string(index=False))
        print(f"\nSaved per-model predictions in: {os.path.abspath(OUTPUT_DIR)}")
        print(f"Saved summary to: {os.path.abspath(summary_path)}")

        excel_path = os.path.join(OUTPUT_DIR, "all_results.xlsx")
        # Build ordered ID list from original row_index to preserve question order
        id_order = (
            all_preds_df[["ID", "row_index", "Question"]]
            .drop_duplicates(subset="ID")
            .sort_values("row_index")
        )
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for rolle in sorted(all_preds_df["rolle"].unique()):
                subset = all_preds_df[all_preds_df["rolle"] == rolle]
                sheet_name = str(rolle)[:31]
                # Build pivot manually to avoid alphabetic sorting by pivot_table
                model_cols = {}
                for model_id, grp in subset.groupby("model"):
                    model_cols[model_id] = grp.set_index("ID")["pred_choice"]
                pivot = pd.DataFrame(model_cols)
                # Join onto id_order to restore original question sequence
                combined = id_order.set_index("ID").join(pivot, how="left")
                combined.to_excel(writer, sheet_name=sheet_name, index=True)
        print(f"Saved Excel overview to: {os.path.abspath(excel_path)}")
    else:
        print("\nNo predictions were generated!")

if __name__ == "__main__":
    main()
