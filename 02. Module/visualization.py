#===========PACKAGES============
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd

#============FUNCTIONS============

# def visualize_performance(reports_dict, title=None, save_path=None):
#     metrics_list = []

#     for label, report in reports_dict.items():
#         report_dict = report
#         metrics = [label,
#                    #report_dict['best_params'],
#                    report_dict['training_accuracy'],
#                    report_dict['testing_accuracy'],
#                    report_dict['runtime']]
#         metrics_list.append(metrics)

#     headers = ['Label', 'Best Params', 'Training Accuracy', 'Testing Accuracy', 'Runtime']
#     df = pd.DataFrame(metrics_list, columns=headers)

#     df.set_index('Label', inplace=True)
#     df = df.round(2)

#     # Create a heatmap-style table using matplotlib
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.axis('off')
#     ax.axis('tight')
#     ax.set_title(title, fontsize=16, fontweight='bold')

#     table = ax.table(cellText=df.values,
#                      colLabels=df.columns,
#                      rowLabels=df.index,
#                      loc='center',
#                      colWidths=[0.15, 0.3, 0.15, 0.15, 0.15],  # Adjust column widths
#                      bbox=[0, 0, 1, 1],  # Specify table bbox to fit within figure
#                      cellLoc='center',
#                      cellLocTxt='center',
#                      cellProps={'wrap': True})

#     # Adjust font size and table size
#     table.auto_set_font_size(False)
#     table.set_fontsize(12)

#     fig.tight_layout()
#     plt.show()

#     # Save the table as a PNG image if a save path is specified
#     if save_path is not None:
#         plt.savefig(save_path, bbox_inches='tight', dpi=150)
#         plt.close()

#     # Return the styled dataframe
#     return table
    

def visualize_performance(reports_dict, title=None, save_path=None):
    metrics_list = []

    for label, report in reports_dict.items():
        report_dict = report
        metrics = [label,
                   report_dict['training_accuracy'],
                   report_dict['testing_accuracy'],
                   report_dict['runtime']]
        metrics_list.append(metrics)

    headers = ['Label', 'Training Accuracy', 'Testing Accuracy', 'Runtime']
    df = pd.DataFrame(metrics_list, columns=headers)

    df.set_index('Label', inplace=True)
    df = df.round(2)

    # Create a heatmap-style table using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.axis('tight')
    ax.set_title(title, fontsize=16, fontweight='bold')

    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2], 
                     bbox=[0, 0, 1, 1]) 

    # Adjust font size and table size
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    fig.tight_layout()
    plt.show()

    # Save the table as a PNG image if a save path is specified
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

    # Return the styled dataframe
    return table

def visualize_classification_reports(reports_dict, title=None, save_path=None):
    # Create an empty list to store the metrics for each report
    metrics_list = []
    
    # Loop through the reports and extract the metrics
    for label, report in reports_dict.items():
        report_dict = report
        metrics = [label,
                   report_dict['accuracy'],
                   report_dict['macro avg']['precision'],
                   report_dict['macro avg']['recall'],
                   report_dict['macro avg']['f1-score']]
        metrics_list.append(metrics)
    
    # Create a pandas dataframe from the metrics list
    headers = ['Label', 'Accuracy', 'Precision', 'Recall', 'F1-score']
    df = pd.DataFrame(metrics_list, columns=headers)
    
    # Set the label column as the index
    df.set_index('Label', inplace=True)
    
    # Format the values to two decimal places
    df = df.round(2)
    
    # Create a heatmap-style table using matplotlib and seaborn
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.axis('tight')
    ax.set_title(title, fontsize=16, fontweight='bold')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc = 'center',
                     loc='center')
 
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    plt.show()

    # Save the table as a PNG image if a save path is specified
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    # Return the styled dataframe
    return table
