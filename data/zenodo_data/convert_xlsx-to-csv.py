import pandas as pd
import yaml
from pathlib import Path

excel_file = "results_combined_footprint_roles.xlsx"
output_dir = Path("csv_exports")
output_dir.mkdir(exist_ok=True)

# Load workbook while preserving sheet order
xls = pd.ExcelFile(excel_file)

# Export each sheet with ordered numbering
for i, sheet_name in enumerate(xls.sheet_names):

    # Read sheet
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    # Remove accidental index columns like "Unnamed: 0"
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    # FAIR-friendly filename
    safe_name = (
        sheet_name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )

    # Master prompts → export as YAML (better for multi-line text)
    if sheet_name.lower() == "master prompts":
        filename = f"{i:02d}_{safe_name}.yaml"
        output_file = output_dir / filename

        prompts = {}
        for col in df.columns:
            prompts[col] = df[col].dropna().tolist()

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(
                prompts,
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

        print(f"Exported (YAML): {output_file}")
        continue

    # Create ordered filename: 00_..., 01_..., etc.
    filename = f"{i:02d}_{safe_name}.csv"

    output_file = output_dir / filename

    # Export as UTF-8 CSV
    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8"
    )

    print(f"Exported: {output_file}")
