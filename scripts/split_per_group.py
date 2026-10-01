"""
Split DICES by annotator or by item for evaluation.
- Split by rater_id: disjoint annotators (train_raters ∩ test_raters = ∅)
- Split by item_id: disjoint items (train_items ∩ test_items = ∅)
"""
import pandas as pd
import sys
from sklearn.model_selection import train_test_split

sys_args = sys.argv[1:]

if len(sys_args) < 3:
    print("Usage: python split_per_group.py <split_by_column> <input_file> <output_file> [--group_level]")
    print("  split_by_column: rater_id (disjoint annotators) or item_id (disjoint items)")
    print("  input_file: path to input CSV")
    print("  output_file: path to output CSV with train/test split column")
    print("  --group_level: (optional) assign whole groups to train or test (disjoint groups)")
    print("                Without it: split within each group (80/20 per group)")
    sys.exit(1)

split_by = sys_args[0]
input_file = sys_args[1]
output_file = sys_args[2]
group_level = "--group_level" in sys_args


def split_per_group(split_by, input_file, output_file, test_size=0.2, random_state=42, group_level=False):
    """
    If group_level=True: disjoint groups (train groups ∩ test groups = ∅).
    If group_level=False: split within each group (80/20 per group).
    """
    df = pd.read_csv(input_file)

    if split_by not in df.columns:
        print(f"Error: Column '{split_by}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    if group_level:
        unique_groups = df[split_by].unique()
        train_groups, test_groups = train_test_split(
            unique_groups, test_size=test_size, random_state=random_state
        )
        train_set = set(train_groups)
        out_df = df.copy()
        out_df["split"] = out_df[split_by].apply(lambda x: "train" if x in train_set else "test")
        print(f"Group-level split by {split_by}: {len(train_groups)} train groups, {len(test_groups)} test groups")
    else:
        results = []
        for group_name, group_df in df.groupby(split_by):
            n = len(group_df)
            if n < 2:
                group_df = group_df.copy()
                group_df["split"] = "train"
                results.append(group_df)
                continue
            train_idx, test_idx = train_test_split(
                group_df.index, test_size=test_size, random_state=random_state
            )
            group_df = group_df.copy()
            group_df["split"] = ""
            group_df.loc[train_idx, "split"] = "train"
            group_df.loc[test_idx, "split"] = "test"
            results.append(group_df)
        out_df = pd.concat(results, ignore_index=True)

    out_df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}: train={sum(out_df['split']=='train')}, test={sum(out_df['split']=='test')}")


if __name__ == "__main__":
    split_per_group(split_by, input_file, output_file, group_level=group_level)
