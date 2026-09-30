"""
Utility functions for saving leaderboard metrics to CSV
"""
import csv
import os


def clean_empty_rows(csv_path):
    """
    Remove any empty rows from the CSV file.
    
    Args:
        csv_path: Path to the CSV file
    """
    if not os.path.exists(csv_path):
        return
    
    # Read all rows, filtering out empty ones
    cleaned_rows = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # Keep row if it has content (not all empty strings)
            if row and any(cell.strip() for cell in row):
                cleaned_rows.append(row)
    
    # Write back only non-empty rows
    if cleaned_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(cleaned_rows)


def save_metrics_to_csv(level, category, metrics_dict, csv_path="leaderboard_metrics.csv"):
    """
    Save or update metrics in the leaderboard CSV file.
    If a metric already exists, it will be updated. Otherwise, it will be added.
    
    Args:
        level: Level number (e.g., "L1", "L2", "L3")
        category: Category short name (e.g., "Wave", "FNO", "Climate", "Fluid")
        metrics_dict: Dictionary of metric_name: metric_value pairs
        csv_path: Path to the CSV file
    """
    # Read existing metrics if file exists
    existing_metrics = {}
    file_exists = os.path.exists(csv_path)
    
    if file_exists:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Metric_Name'] and row['Metric_Value']:
                    existing_metrics[row['Metric_Name']] = row['Metric_Value']
    
    # Update or add new metrics
    for metric_name, metric_value in metrics_dict.items():
        # Format: L1_Wave_Validation_RMSE (remove spaces and parentheses)
        formatted_name = f"{level}_{category}_{metric_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        existing_metrics[formatted_name] = metric_value
    
    # Write all metrics back to file
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric_Name', 'Metric_Value'])
        
        # Write all metrics in sorted order for consistency
        for metric_name in sorted(existing_metrics.keys()):
            writer.writerow([metric_name, existing_metrics[metric_name]])
    
    # Clean up any empty rows that might have been created
    clean_empty_rows(csv_path)
    
    print(f"\n✅ Metrics saved to {csv_path}")

