import json

from matplotlib import pyplot as plt


def plot_test_results(true_vals, pred_vals):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.4)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    plt.title("Validation Predictions vs True Scores")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def score_dis(df):
    plt.figure(figsize=(8, 5))
    plt.hist(df['visual_score'], edgecolor='black')

    # 3. Add titles and labels
    plt.title('Distribution of Visual Scores')
    plt.xlabel('Visual Score')
    plt.ylabel('Count')

    # 4. Show grid for easier reading
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_graphs(df, table_name_score):
    """
    For each row in df, parses JSON strings in 'head_norm_y' and 'ocr_norm_y',
    then plots both series on the same figure and annotates the visual score in the title.
    """
    df_copy = df.copy()
    df_copy['head_norm_y'] = df_copy['head_norm_y'].apply(json.loads)
    df_copy['ocr_norm_y'] = df_copy['ocr_norm_y'].apply(json.loads)

    for idx, row in df_copy.iterrows():
        head = row['head_norm_y']
        ocr = row['ocr_norm_y']
        predicted_score = row[table_name_score]
        visual_score = row['visual_score']

        plt.figure()
        plt.plot(head, label='Head')
        plt.plot(ocr, label='OCR')
        plt.title(f'Row {idx} – Visual Score: {visual_score:.2f} Predicted Score: {predicted_score}')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.tight_layout()
        plt.show()