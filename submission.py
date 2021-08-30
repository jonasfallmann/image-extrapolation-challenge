import dill as pkl
import os

import torch
import tqdm
from plotting import plot
import numpy as np
import architecture


def evaluate_model(inputs: str, model_path: str, out_path: str, plot_path: str, mean: float, std: float):
    abs_inputs_path = os.path.abspath(inputs)
    with open(abs_inputs_path, 'rb') as infile:
        inputs = pkl.load(infile)

    image_arrays = inputs['input_arrays']
    known_arrays = inputs['known_arrays']
    borders_x = inputs['borders_x']
    borders_y = inputs['borders_y']

    abs_out_path = os.path.abspath(out_path)
    abs_model_path = os.path.abspath(model_path)
    net = torch.load(abs_model_path)
    device = torch.device('cuda:0')

    plotpath = os.path.abspath(plot_path)

    net.to(device)
    outputs = []
    update_progess_bar = tqdm.tqdm(total=len(image_arrays), position=0)  # progressbar

    for idx, elem in enumerate(image_arrays):
        input_d = torch.zeros(1, 2, 90, 90)
        img = (elem / 255.0)
        input_d[0][0] = torch.tensor(img)
        input_d[0][1] = torch.tensor(known_arrays[idx])
        input_d = input_d.to(device)
        output, flat = net(input_d)
        length = 90 * 90 - (90 - borders_x[idx][0] - borders_x[idx][1]) * (90 - borders_y[idx][0] - borders_y[idx][1])
        trimmed = flat[0][:length] * 255
        outputs.append(trimmed.detach().cpu().numpy().astype(np.uint8))
        plot(input_d.detach().cpu().numpy(), output.detach().cpu().numpy(), 1, 1,
             plotpath, idx)
        update_progess_bar.update()

    with open(abs_out_path, 'wb') as out:
        pkl.dump(outputs, out)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, help="Path to input pickle file")
    parser.add_argument("--model", type=str, help="Path to input pickle file")
    parser.add_argument("--out", type=str, help="Path to input pickle file")
    parser.add_argument("--plots", type=str, help="Path to input pickle file")
    parser.add_argument("--mean", type=float, help="")
    parser.add_argument("--std", type=float, help="")
    args = parser.parse_args()
    evaluate_model(args.inputs, args.model, args.out, args.plots, args.mean, args.std)
