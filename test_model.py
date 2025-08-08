import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
                           precision_recall_curve, average_precision_score,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# === SETUP ===
test_dir = 'split/test'  # Path folder test
img_size = (224, 224)    # Ganti sesuai ukuran input model
batch_size = 16
model_path = 'model_final.keras'  # Path model

def create_output_directory():
    """Create output directory for saving plots"""
    output_dir = 'evaluation_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_and_prepare_data():
    """Load and prepare test data"""
    print("Loading test data...")
    
    # Create data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Important for proper evaluation
    )
    
    print(f"Found {test_generator.samples} test samples")
    print(f"Classes: {list(test_generator.class_indices.keys())}")
    
    return test_generator

def load_trained_model():
    """Load the trained model"""
    print("Loading model...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_predictions(model, test_generator):
    """Get model predictions"""
    print("Making predictions...")
    
    # Get predictions
    pred_probs = model.predict(test_generator, verbose=1)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    
    print("Predictions completed!")
    return pred_probs, pred_classes, true_classes, class_labels

def plot_confusion_matrix(true_classes, pred_classes, class_labels, output_dir):
    """Plot and save confusion matrix"""
    print("Creating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(true_classes, pred_classes)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot count matrix
    sns.heatmap(cm, annot=True, fmt="d", 
               xticklabels=class_labels, yticklabels=class_labels, 
               cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix (Count)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=12)
    axes[0].set_ylabel("True Label", fontsize=12)
    
    # Calculate and plot percentage matrix
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(cm_percent, annot=True, fmt=".1f", 
               xticklabels=class_labels, yticklabels=class_labels, 
               cmap="Blues", ax=axes[1])
    axes[1].set_title("Confusion Matrix (Percentage)", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Predicted Label", fontsize=12)
    axes[1].set_ylabel("True Label", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def print_classification_metrics(true_classes, pred_classes, class_labels):
    """Print detailed classification metrics"""
    print("\n" + "="*50)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*50)
    
    # Overall metrics
    accuracy = accuracy_score(true_classes, pred_classes)
    precision_macro = precision_score(true_classes, pred_classes, average='macro')
    recall_macro = recall_score(true_classes, pred_classes, average='macro')
    f1_macro = f1_score(true_classes, pred_classes, average='macro')
    
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Macro-averaged Precision: {precision_macro:.4f}")
    print(f"Macro-averaged Recall: {recall_macro:.4f}")
    print(f"Macro-averaged F1-score: {f1_macro:.4f}")
    
    print("\nPer-Class Classification Report:")
    print("-" * 50)
    print(classification_report(true_classes, pred_classes, target_names=class_labels, digits=4))
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Class': class_labels,
        'Precision': precision_score(true_classes, pred_classes, average=None),
        'Recall': recall_score(true_classes, pred_classes, average=None),
        'F1-Score': f1_score(true_classes, pred_classes, average=None)
    })
    
    print("\nMetrics Summary Table:")
    print("-" * 50)
    print(metrics_df.round(4))
    
    return metrics_df

def plot_roc_curves(true_classes, pred_probs, class_labels, output_dir):
    """Plot ROC curves for multi-class classification"""
    print("Creating ROC curves...")
    
    n_classes = len(class_labels)
    
    # Binarize the true labels
    y_true_bin = label_binarize(true_classes, classes=range(n_classes))
    
    # If binary classification, reshape
    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
    
    # Compute ROC curve for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    # Colors for plotting
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 
                   'purple', 'brown', 'pink', 'gray', 'olive'])
    
    plt.figure(figsize=(12, 10))
    
    for i, color in zip(range(n_classes), colors):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_labels[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Compute micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), pred_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle='--', lw=2,
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})')
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Multi-Class Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fpr, tpr, roc_auc

def plot_precision_recall_curves(true_classes, pred_probs, class_labels, output_dir):
    """Plot Precision-Recall curves"""
    print("Creating Precision-Recall curves...")
    
    n_classes = len(class_labels)
    
    # Binarize the true labels
    y_true_bin = label_binarize(true_classes, classes=range(n_classes))
    
    # If binary classification, reshape
    if n_classes == 2:
        y_true_bin = np.column_stack([1 - y_true_bin, y_true_bin])
    
    # Compute PR curve for each class
    precision = {}
    recall = {}
    avg_precision = {}
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 
                   'purple', 'brown', 'pink', 'gray', 'olive'])
    
    plt.figure(figsize=(12, 10))
    
    for i, color in zip(range(n_classes), colors):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], 
                                                           pred_probs[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], 
                                                  pred_probs[:, i])
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'{class_labels[i]} (AP = {avg_precision[i]:.3f})')
    
    # Compute micro-average PR curve
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), 
                                                             pred_probs.ravel())
    avg_precision_micro = average_precision_score(y_true_bin, pred_probs, 
                                                 average='micro')
    
    plt.plot(recall_micro, precision_micro, color='deeppink', linestyle='--', lw=2,
            label=f'Micro-average (AP = {avg_precision_micro:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves for Multi-Class Classification', 
             fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    plt.savefig(f'{output_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return precision, recall, avg_precision

def plot_class_distribution(true_classes, class_labels, output_dir):
    """Plot class distribution in test set"""
    print("Creating class distribution plot...")
    
    # Count samples per class
    unique, counts = np.unique(true_classes, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_labels)), 
                   [counts[i] if i in unique else 0 for i in range(len(class_labels))],
                   color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_labels)), class_labels, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_summary(metrics_df, roc_auc, avg_precision, output_dir):
    """Save comprehensive results summary"""
    print("Saving results summary...")
    
    # Create summary dictionary
    summary = {
        'Model Path': model_path,
        'Test Directory': test_dir,
        'Image Size': img_size,
        'Batch Size': batch_size,
        'Number of Classes': len(metrics_df),
        'Overall Accuracy': metrics_df['F1-Score'].mean(),  # Using mean F1 as proxy
    }
    
    # Add per-class metrics
    for i, row in metrics_df.iterrows():
        class_name = row['Class']
        summary[f'{class_name}_Precision'] = row['Precision']
        summary[f'{class_name}_Recall'] = row['Recall']
        summary[f'{class_name}_F1'] = row['F1-Score']
        if i in roc_auc:
            summary[f'{class_name}_AUC'] = roc_auc[i]
        if i in avg_precision:
            summary[f'{class_name}_AP'] = avg_precision[i]
    
    # Save to CSV
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f'{output_dir}/evaluation_summary.csv', index=False)
    
    # Save detailed metrics
    metrics_df.to_csv(f'{output_dir}/detailed_metrics.csv', index=False)
    
    print(f"Results saved to {output_dir}/")

def main():
    """Main evaluation function"""
    print("Starting Model Evaluation...")
    print("=" * 50)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Load data and model
    test_generator = load_and_prepare_data()
    model = load_trained_model()
    
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    # Get predictions
    pred_probs, pred_classes, true_classes, class_labels = get_predictions(model, test_generator)
    
    # Create visualizations and metrics
    cm = plot_confusion_matrix(true_classes, pred_classes, class_labels, output_dir)
    metrics_df = print_classification_metrics(true_classes, pred_classes, class_labels)
    fpr, tpr, roc_auc = plot_roc_curves(true_classes, pred_probs, class_labels, output_dir)
    precision, recall, avg_precision = plot_precision_recall_curves(true_classes, pred_probs, class_labels, output_dir)
    plot_class_distribution(true_classes, class_labels, output_dir)
    
    # Save comprehensive results
    save_results_summary(metrics_df, roc_auc, avg_precision, output_dir)
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETED!")
    print(f"All results saved to: {output_dir}/")
    print("=" * 50)

if __name__ == "__main__":
    main()