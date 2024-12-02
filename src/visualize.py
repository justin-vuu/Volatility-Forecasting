import matplotlib.pyplot as plt
import seaborn as sns

def plot_train_val_metrics(model_hist):
    # Plot MSE and RMSPE curves
    sns.set_context("talk", font_scale=1.3)
    fig, ax = plt.subplots(figsize=(15,6))

    ax.plot(model_hist.history['loss'], label='Training MSE', lw=2, color='lightblue')
    ax.plot(model_hist.history['val_loss'], label='Validation MSE', lw=2, color='darkblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')

    ax2 = ax.twinx()
    ax2.plot(model_hist.history['rmspe'], label='Training RMSPE', lw=2, color='pink')
    ax2.plot(model_hist.history['val_rmspe'], label='Validation RMSPE', lw=2, color='red')
    ax2.set_ylabel('RMSPE')

    ax.legend(loc='best')
    ax2.legend(loc='best')
    plt.title('Training vs. Validation Learning Curves')
    plt.show()

def plot_model_predictions(y_true, y_pred, title="Model Prediction"):
    sns.set_context("paper", font_scale=1.7)
    plt.figure(figsize=(18,7))
    plt.plot(y_true, label='True', color='blue', lw=2)
    plt.plot(y_pred, label='Predicted', color='orange', lw=2.5)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()
