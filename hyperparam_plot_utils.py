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