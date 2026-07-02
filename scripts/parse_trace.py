import argparse
import os
import sys
import pandas as pd


def format_kernel_name(name):
    for pattern in ["basic_parallel_for<", "ndrange_parallel_for<"]:
        if pattern in name:
            start = name.find(pattern) + len(pattern)
            sub = name[start:]
            paren = sub.find("(")
            if paren != -1:
                return sub[:paren].strip()
            return sub.strip()

    if len(name) > 60:
        return name[:57] + "..."
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a rocprofv3 kernel trace CSV for register pressure."
    )
    parser.add_argument("csv_path", type=str, help="Path to the kernel_trace.csv file")
    args = parser.parse_args()

    # Verify file existence
    if not os.path.exists(args.csv_path):
        print(f"Error: File '{args.csv_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Read the CSV
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Ensure required columns exist
    required_cols = [
        "Start_Timestamp",
        "End_Timestamp",
        "Kernel_Name",
        "VGPR_Count",
        "SGPR_Count",
        "Scratch_Size",
        "LDS_Block_Size",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Error: The provided CSV is missing profiling columns: {missing_cols}",
            file=sys.stderr,
        )
        print(
            "Make sure you passed the '*_kernel_trace.csv' file and not '*_agent_info.csv'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Calculate duration in milliseconds
    df["Duration_Ms"] = (df["End_Timestamp"] - df["Start_Timestamp"]) / 1e6

    # Group and aggregate
    summary = (
        df.groupby("Kernel_Name")
        .agg(
            Calls=("Kernel_Name", "count"),
            Total_Time_Ms=("Duration_Ms", "sum"),
            Avg_Time_Ms=("Duration_Ms", "mean"),
            Avg_VGPR=("VGPR_Count", "mean"),
            Avg_SGPR=("SGPR_Count", "mean"),
            Max_Scratch_Bytes=("Scratch_Size", "max"),
            Avg_LDS_Bytes=("LDS_Block_Size", "mean"),
        )
        .sort_values(by="Total_Time_Ms", ascending=False)
        .reset_index()
    )

    summary["Short_Name"] = summary["Kernel_Name"].apply(format_kernel_name)

    # Output layout definition
    header = f"{'Kernel Name':<60} | {'Calls':>5} | {'Total (ms)':>10} | {'Avg (ms)':>10} | {'VGPR':>4} | {'SGPR':>4} | {'Scratch':>8} | {'LDS':>6}"
    separator = "-" * len(header)

    print("=== TOP KERNELS BY TOTAL EXECUTION TIME ===")
    print(header)
    print(separator)
    for _, row in summary.head(10).iterrows():
        print(
            f"{row['Short_Name']:<60} | "
            f"{int(row['Calls']):>5} | "
            f"{row['Total_Time_Ms']:>10.2f} | "
            f"{row['Avg_Time_Ms']:>10.3f} | "
            f"{int(row['Avg_VGPR']):>4} | "
            f"{int(row['Avg_SGPR']):>4} | "
            f"{int(row['Max_Scratch_Bytes']):>8} | "
            f"{int(row['Avg_LDS_Bytes']):>6}"
        )

    print("\n=== KERNELS WITH REGISTER SPILLS (SCRATCH > 0) ===")
    spills = summary[summary["Max_Scratch_Bytes"] > 0]
    if spills.empty:
        print("No register spills detected! Your kernels fit entirely in registers.")
    else:
        print(header)
        print(separator)
        for _, row in spills.iterrows():
            print(
                f"{row['Short_Name']:<60} | "
                f"{int(row['Calls']):>5} | "
                f"{row['Total_Time_Ms']:>10.2f} | "
                f"{row['Avg_Time_Ms']:>10.3f} | "
                f"{int(row['Avg_VGPR']):>4} | "
                f"{int(row['Avg_SGPR']):>4} | "
                f"{int(row['Max_Scratch_Bytes']):>8} | "
                f"{int(row['Avg_LDS_Bytes']):>6}"
            )


if __name__ == "__main__":
    main()
