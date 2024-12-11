import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import optuna.visualization as vis

from optuna.study import Study


def plot_param_importances_grid(
    study: Study,
    hyperparams_per_row: int,
    subplot_height: int,
    subplot_width: int,
):
    """Plot parameter importances in a grid

    :param study: Optuna study
    :param hyperparams_per_row: Number of hyperparameters per row
    :param subplot_height: Subplot height
    :param subplot_width: Subplot width
    """
    # Calculate the number of rows needed
    n_rows = len(study.best_trial.params) // hyperparams_per_row + (
        len(study.best_trial.params) % hyperparams_per_row > 0
    )
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=hyperparams_per_row,
        figsize=(subplot_width * hyperparams_per_row, subplot_height * n_rows),
    )
    # Flatten the axes array for easier indexing (handles single-row case as well)
    axs = axs.flatten()
    for i, (key, value) in enumerate(study.best_trial.params.items()):
        ax = axs[i]
        # Scatter plot of the parameter values vs. the objective value
        ax.scatter(
            study.trials_dataframe()["params_" + key],
            study.trials_dataframe()["value"],
        )
        # Highlight the best trial value with a dashed red line
        ax.axhline(study.best_trial.value, color="r", linestyle="--")
        # Display the best trial value as text on the plot
        ax.text(
            0.5,
            0.9,
            f"Best: {study.best_trial.value:.4f}",
            color="red",
            fontsize=10,
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        # Titles and labels
        ax.set_title(key)
        ax.set_xlabel(key)
        ax.set_ylabel("Objective Value")
        ax.grid()
    # Hide unused subplots
    for j in range(len(study.best_trial.params), len(axs)):
        axs[j].axis("off")
    plt.tight_layout()
    plt.show()


def plot_rank(study: Study, n_params: int, width: int, height: int):
    """Plot relationship between parameters and objective value

    :param study: Optuna study
    :param n_params: Number of important parameters to plot
    :param width: Plot width
    :param height: Plot height
    """
    # Get parameters sorted by the importance values
    importances = optuna.importance.get_param_importances(study)
    params_sorted = list(importances.keys())
    # Plot
    fig = optuna.visualization.plot_rank(study, params=params_sorted[:n_params])
    #Make the fig larger
    fig.update_layout(width=width, height=height)
    fig.show()


def plot_param_importances(study: Study):
    """Wrapper for optuna.visualization.plot_param_importances

    :param study: Optuna study
    """
    fig = vis.plot_param_importances(study)
    fig.show()


def plot_optimization_history(study: Study):
    """Wrapper for optuna.visualization.plot_optimization_history

    :param study: Optuna study
    """
    fig = vis.plot_optimization_history(study)
    fig.show()


def plot_parallel_coordinate(study: Study):
    """Wrapper for optuna.visualization.plot_parallel_coordinate

    :param study: Optuna study
    """
    fig = vis.plot_parallel_coordinate(study)
    fig.show()


def plot_hyperparam_correlation(study: Study):
    """Plot correlation between hyperparameters and objective

    :param study: Optuna study
    """
    df = study.trials_dataframe().filter(like="params_")
    df["objective"] = study.trials_dataframe()["value"]
    corr = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlación entre Hiperparámetros y Objetivo")
    plt.show()



    """
    Creates a visualization of rating distribution with bar plot and pie chart for two datasets.
    
    Parameters:
    -----------
    dataset : pandas DataFrame
        DataFrame containing a 'rating' column
    dataset_original : pandas DataFrame
        Original DataFrame containing a 'rating' column
    figsize : tuple, optional
        Figure size for the plots (width, height)
    palette : str, optional
        Color palette to use for the plots
    
    Returns:
    --------
    dict
        Dictionary containing the numerical summaries for both datasets
    """
    
    # Create a figure with four subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Calculate total records for both datasets
    total_records = len(dataset)
    total_records_original = len(dataset_original)
    
    # 1. Bar plot with seaborn for the first dataset
    sns.countplot(data=dataset, x='rating', ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title(f'Rating Distribution (Bar Plot) - Cleaned\nTotal Records: {total_records:,}')
    axes[0, 0].set_xlabel('Rating')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Pie chart with matplotlib for the first dataset
    rating_counts = dataset['rating'].value_counts()
    colors = sns.color_palette(palette, n_colors=len(rating_counts))
    axes[0, 1].pie(rating_counts, 
                   labels=rating_counts.index, 
                   autopct='%1.1f%%', 
                   colors=colors)
    axes[0, 1].set_title('Rating Distribution (Pie Chart) - Cleaned')
    
    # 3. Bar plot with seaborn for the original dataset
    sns.countplot(data=dataset_original, x='rating', ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title(f'Rating Distribution (Bar Plot) - Original\nTotal Records: {total_records_original:,}')
    axes[1, 0].set_xlabel('Rating')
    axes[1, 0].set_ylabel('Count')
    
    # 4. Pie chart with matplotlib for the original dataset
    rating_counts_original = dataset_original['rating'].value_counts()
    colors_original = sns.color_palette(palette, n_colors=len(rating_counts_original))
    axes[1, 1].pie(rating_counts_original, 
                   labels=rating_counts_original.index, 
                   autopct='%1.1f%%', 
                   colors=colors_original)
    axes[1, 1].set_title('Rating Distribution (Pie Chart) - Original')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    # Calculate numerical summaries for both datasets
    counts = dataset['rating'].value_counts().sort_index()
    percentages = dataset['rating'].value_counts(normalize=True).sort_index().mul(100).round(1)
    
    counts_original = dataset_original['rating'].value_counts().sort_index()
    percentages_original = dataset_original['rating'].value_counts(normalize=True).sort_index().mul(100).round(1)
    
    # Print numerical summary for both datasets
    print("\nNumerical Summary - Cleaned:")
    print(counts)
    print("\nPercentage Distribution - Cleaned:")
    print(percentages)
    
    print("\nNumerical Summary - Original:")
    print(counts_original)
    print("\nPercentage Distribution - Original:")
    print(percentages_original)
    
    # Return the summaries as a dictionary
    return {
        'cleaned': {
            'counts': counts,
            'percentages': percentages
        },
        'original': {
            'counts': counts_original,
            'percentages': percentages_original
        }
    }