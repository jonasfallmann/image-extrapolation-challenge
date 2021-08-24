import os
from matplotlib import pyplot as plt


def plot(inputs, predictions, path, update):
    """Plotting the inputs, targets, and predictions to file `path`"""
    os.makedirs(path, exist_ok=True)
    fig, ax = plt.subplots(2, 2)
    ax[0, 1].remove()

    for i in range(len(inputs)):
        ax[0, 0].clear()
        ax[0, 0].set_title('input')
        ax[0, 0].imshow(inputs[i, 0], cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=255)
        ax[0, 0].set_axis_off()
        # ax[0, 1].clear()
        # ax[0, 1].set_title('targets')
        # ax[0, 1].imshow(targets[i, 0], cmap=plt.cm.gray, interpolation='none')
        # ax[0, 1].set_axis_off()
        ax[1, 0].clear()
        ax[1, 0].set_title('predictions')
        ax[1, 0].imshow(predictions[i, 0], cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=255)
        ax[1, 0].set_axis_off()

        ax[1, 1].clear()
        ax[1, 1].set_title('merged')
        merged = inputs[i][0].copy()
        mask = inputs[i][1]
        asd = predictions[i][0][mask == 0]
        merged[mask == 0] = predictions[i][0][mask == 0]
        ax[1, 1].imshow(merged, cmap=plt.cm.gray, interpolation='none', vmin=0, vmax=255)
        ax[1, 1].set_axis_off()

        fig.tight_layout()
        fig.savefig(os.path.join(path, f"{update:07d}_{i:02d}.png"), dpi=1000)

    plt.close(fig)
    del fig
