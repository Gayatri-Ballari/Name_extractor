import validation  # Importing existing script
import pytest    # for the test cases
import pandas as pd #for reading Csv
import matplotlib.pyplot as plt #visualization
import csv
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def save_metrics_to_csv(file_path="evaluation_results.csv"):
    """Saves the collected evaluation metrics into a CSV file."""
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        fieldnames = ["Text","Ground Truth","Predictions" ,"Precision", "Recall", "F1-score", "Fuzzy Score"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()  # Write column headers
        writer.writerows(all_metrics)  # Write all collected metrics
    print(f"Evaluation results saved to {file_path}")

# Store all metrics for visualization
all_metrics = []

def fuzzy_match(predictions, ground_truth, threshold=50):
    """Applies fuzzy matching to compare extracted company names with ground truth."""
    matched_predictions = []
    fuzzy_scores = []
    for pred in predictions:
        match, score = process.extractOne(pred, ground_truth) if ground_truth else (None, 0)
        print (f'here us the match value {match}')
        if score >= threshold:
            matched_predictions.append(match)
            fuzzy_scores.append(score)
    return matched_predictions, fuzzy_scores

# NER Evaluation Function
def evaluate_ner(predictions, ground_truth):
    """Evaluates Named Entity Recognition (NER) model predictions against ground truth labels."""
    predictions, fuzzy_scores = fuzzy_match(predictions, ground_truth)
    true_positives = sum(1 for pred in predictions if pred in ground_truth)
    false_positives = sum(1 for pred in predictions if pred not in ground_truth)
    false_negatives = sum(1 for gt in ground_truth if gt not in predictions)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    avg_fuzzy_score = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0

    return {"Precision": precision, "Recall": recall, "F1-score": f1, "Fuzzy Score": avg_fuzzy_score}

# Function to Extract and Evaluate
def process_text(text, ground_truth):
    """Extracts company names from text and evaluates against ground truth."""
    predictions = validation.extract_list_from_text(validation.extract_company_names(text))
    metrics = evaluate_ner(predictions, ground_truth)

    # Store the metrics with additional information for visualization
    all_metrics.append({
        "Text": text,
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-score": metrics["F1-score"],
        "Fuzzy Score": metrics["Fuzzy Score"]
    })

    print(f"Text: {text}")
    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Evaluation Metrics: {metrics}")
    print("-" * 80)

    return metrics

# To Load the Synthetic Data from CSV
def load_test_cases_from_csv(file_path="Company extractor validation data - Sheet1.csv"):
    """Loads test cases from a CSV file."""
    df = pd.read_csv(file_path)
    df["ground_truth"] = df["ground_truth"].apply(lambda x: x.strip("[]").replace("'", "").split(", ") if isinstance(x, str) else [])
    return df.to_dict("records")  # Convert DataFrame to list of dictionaries

# To Run the Synthetic Tests
def run_synthetic_tests():
    """Runs synthetic tests loaded from a CSV file."""
    test_cases = load_test_cases_from_csv()
    for test in test_cases:
        text = test["text"]
        ground_truth = test["ground_truth"]
        process_text(text, ground_truth)

# Function to Calculate Mean and Median Fuzzy Score
def calculate_fuzzy_score_stats():
    """Calculates and prints the mean and median fuzzy scores."""
    df_metrics = pd.DataFrame(all_metrics)
    mean_fuzzy_score = df_metrics["Fuzzy Score"].mean()
    median_fuzzy_score = df_metrics["Fuzzy Score"].median()
    print(f"Mean Fuzzy Score: {mean_fuzzy_score}")
    print(f"Median Fuzzy Score: {median_fuzzy_score}")

def visualize_metrics():
    """Visualizes the metrics with metrics on the x-axis against mean and median."""
    # Convert collected metrics into a DataFrame
    df_metrics = pd.DataFrame(all_metrics)
    
    # Compute the mean and median of each metric
    mean_metrics = df_metrics[["Precision", "Recall", "F1-score"]].mean()
    median_metrics = df_metrics[["Precision", "Recall", "F1-score"]].median()
    
    # Set the figure size
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot a bar chart with metrics on the x-axis
    mean_metrics.plot(kind="bar", ax=ax, color=['blue', 'green', 'red'], alpha=0.6, label='Mean')
    
    # Plot the mean and median metrics as separate lines
    ax.axhline(mean_metrics["Precision"], color='blue', linestyle='--', label='Mean Precision')
    ax.axhline(mean_metrics["Recall"], color='green', linestyle='--', label='Mean Recall')
    ax.axhline(mean_metrics["F1-score"], color='red', linestyle='--', label='Mean F1-score')

    ax.axhline(median_metrics["Precision"], color='blue', linestyle=':', label='Median Precision')
    ax.axhline(median_metrics["Recall"], color='green', linestyle=':', label='Median Recall')
    ax.axhline(median_metrics["F1-score"], color='red', linestyle=':', label='Median F1-score')
    
    # Add title and labels
    plt.title('Mean and Median Evaluation Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Metric Values')
    plt.xticks(rotation=0)  # Keep metric labels readable
    plt.legend(title="Statistics")
    
    plt.tight_layout()
    plt.show()


def visualize_fuzzy_score_distribution():
    """Visualizes the distribution of fuzzy scores with mean and median."""
    df_metrics = pd.DataFrame(all_metrics)
    plt.figure(figsize=(10, 6))

    # Plot the distribution of fuzzy scores
    plt.hist(df_metrics["Fuzzy Score"], bins=20, color='skyblue', edgecolor='black', alpha=0.7)

    # Mean and Median lines
    mean_fuzzy_score = df_metrics["Fuzzy Score"].mean()
    median_fuzzy_score = df_metrics["Fuzzy Score"].median()
    plt.axvline(mean_fuzzy_score, color='red', linestyle='--', label=f'Mean: {mean_fuzzy_score:.2f}')
    plt.axvline(median_fuzzy_score, color='green', linestyle=':', label=f'Median: {median_fuzzy_score:.2f}')

    plt.title('Distribution of Fuzzy Scores with Mean and Median')
    plt.xlabel('Fuzzy Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Histogram for Mean and Median
    plt.figure(figsize=(8, 5))
    plt.bar(['Mean', 'Median'], [mean_fuzzy_score, median_fuzzy_score], color=['red', 'green'])
    plt.title('Mean and Median of Fuzzy Scores')
    plt.ylabel('Fuzzy Score')
    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    print("Running Synthetic Data Tests from csv...\n")
    run_synthetic_tests()
    print("Saving Metrics to CSV...\n")
    save_metrics_to_csv()
    print("Calculating Mean and Median Fuzzy Score...\n")
    calculate_fuzzy_score_stats()
    print("Visualizing the Validation Metrics...\n")
    visualize_metrics()
    print("distribution of Fuzzy score")
    visualize_fuzzy_score_distribution()
