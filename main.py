import torch
import os
import numpy as np
import tqdm

from torch.utils.tensorboard import SummaryWriter
from architecture import ImageExtrapolationCNN
from datasets import ImageExtrapolation, collate_fn, ImageDataset
from normalization import NormalizationProvider
from plotting import plot


def main(results_path, network_config: dict, dataset_path: str, learningrate: int = 1e-3, weight_decay: float = 1e-5,
         n_updates: int = int(1e5), mean: float = 0.0, std: float = 0.0,
         device: torch.device = torch.device('cuda:0')):
    plotpath = os.path.join(results_path, 'plots')
    os.makedirs(plotpath, exist_ok=True)

    abs_dataset_path = os.path.abspath(dataset_path)
    dataset = ImageDataset(abs_dataset_path)

    # create Subsets
    trainingset_base = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (3 / 5))))
    validationset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (3 / 5)),
                                                                       int(len(dataset) * (4 / 5))))
    testset = torch.utils.data.Subset(dataset, indices=np.arange(int(len(dataset) * (4 / 5)),
                                                                 len(dataset)))
    # Create normalization provider
    normalization_provider = NormalizationProvider(mean, std)
    if mean == 0.0 and std == 0.0:
        normalization_provider.from_dataset(trainingset_base)

    # Create Datasets
    trainingset = ImageExtrapolation(trainingset_base, normalization_provider)
    validationset = ImageExtrapolation(validationset, normalization_provider)
    testset = ImageExtrapolation(testset, normalization_provider)
    trainingset_augment = ImageExtrapolation(trainingset_base, normalization_provider, True, False)
    trainingset_crop_randomly = ImageExtrapolation(trainingset_base, normalization_provider, True, True)

    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=5, shuffle=True,
                                              num_workers=5,
                                              collate_fn=collate_fn)
    trainloader_augmented = torch.utils.data.DataLoader(trainingset_augment, batch_size=5, shuffle=True,
                                                        num_workers=5,
                                                        collate_fn=collate_fn)
    trainloader_crop_randomly = torch.utils.data.DataLoader(trainingset_crop_randomly, batch_size=5, shuffle=True,
                                                            num_workers=5,
                                                            collate_fn=collate_fn)

    valloader = torch.utils.data.DataLoader(validationset, batch_size=2, shuffle=False, num_workers=8,
                                            collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=False, num_workers=8,
                                             collate_fn=collate_fn)

    writer = SummaryWriter(log_dir=os.path.join(results_path, 'tensorboard'))

    # create Net
    net = ImageExtrapolationCNN()
    net.to(device)

    # Loss
    mse = torch.nn.MSELoss()

    # OPtimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learningrate, weight_decay=weight_decay)

    print_stats_at = 1e2  # print status to tensorboard every x updates
    plot_at = 5e3  # plot every x updates
    validate_at = 5e3  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progess_bar = tqdm.tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)  # progressbar

    torch.save(net, os.path.join(results_path, 'best_model.pt'))
    update = 0
    no_last_losses = 10
    last_losses = [0] * no_last_losses
    while update < n_updates:
        augmented_iterator = iter(trainloader_augmented)
        crop_iterator = iter(trainloader_crop_randomly)
        for data in trainloader:
            inputs_augmented, targets_augmented = next(augmented_iterator)
            inputs_crop, targets_crop = next(crop_iterator)
            inputs, targets = data

            inputs = torch.cat((inputs_augmented, inputs_crop, inputs))
            targets = torch.cat((targets_augmented, targets_crop, targets))

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Set gradients to 0
            optimizer.zero_grad()

            outputs, flat = net(inputs)

            loss = mse(flat.to(device), targets)
            loss.backward()
            optimizer.step()

            last_losses[update % no_last_losses] = loss.detach().cpu().numpy() * 65025

            # Print current status and score
            if update % print_stats_at == 0 and update > 0:
                writer.add_scalar(tag="training/loss",
                                  scalar_value=np.mean(last_losses).astype(float).item(),
                                  global_step=update)

            # Plot output
            if update % plot_at == 0 and update > 0:
                plot(inputs.detach().cpu().numpy(), outputs.detach().cpu().numpy(), mean, std,
                     plotpath, update)

            # Evaluate model on validation set
            if update % validate_at == 0 and update > 0:
                net.eval()
                val_loss = evaluate_model(net, dataloader=valloader, device=device)
                net.train()
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss.cpu(), global_step=update)
                # Add weights as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/param_{i}', values=param.cpu(),
                                         global_step=update)
                # Add gradients as arrays to tensorboard
                for i, param in enumerate(net.parameters()):
                    writer.add_histogram(tag=f'validation/gradients_{i}',
                                         values=param.grad.cpu(),
                                         global_step=update)
                # Save best model for early stopping
                if best_validation_loss > val_loss:
                    best_validation_loss = val_loss
                    torch.save(net, os.path.join(results_path, 'best_model.pt'))

            update_progess_bar.set_description(f"loss: {loss * 65025:7.5f}", refresh=True)
            update_progess_bar.update()

            # Increment update counter, exit if maximum number of updates is reached
            update += 1
            if update >= n_updates:
                break
    update_progess_bar.close()
    print('Finished Training!')

    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    net = torch.load(os.path.join(results_path, 'best_model.pt'))
    test_loss = evaluate_model(net, dataloader=testloader, device=device)
    val_loss = evaluate_model(net, dataloader=valloader, device=device)
    train_loss = evaluate_model(net, dataloader=trainloader, device=device)

    print(f"Scores:")
    print(f"test loss: {test_loss}")
    print(f"validation loss: {val_loss}")
    print(f"training loss: {train_loss}")

    # Write result to file
    with open(os.path.join(results_path, 'results.txt'), 'w') as fh:
        print(f"Scores:", file=fh)
        print(f"test loss: {test_loss}", file=fh)
        print(f"validation loss: {val_loss}", file=fh)
        print(f"training loss: {train_loss}", file=fh)
    return 0


def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`"""
    # Define a loss (mse loss)
    mse = torch.nn.MSELoss()
    # We will accumulate the mean loss in variable `loss`
    loss = torch.tensor(0., device=device)
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for data in tqdm.tqdm(dataloader, desc="scoring", position=0):
            # Get a sample and move inputs and targets to device
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get outputs for network
            outputs, flat = model(inputs)
            flat = flat.to(device)

            # Here we could clamp the outputs to the minimum and maximum values of inputs for better performance

            # Calculate mean mse loss over all samples in dataloader (accumulate mean losses in `loss`)
            loss += (torch.stack([mse(output, target) for output, target in zip(flat, targets)]).sum()
                     / len(dataloader.dataset))
    return loss * 65025


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='path to the config', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as cf:
        config = json.load(cf)
    main(**config)
