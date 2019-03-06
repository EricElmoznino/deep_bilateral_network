import torch
from torch.nn import functional as F

import configuration as conf
import utils


def compute_psnr(predictions, targets):
    mse = F.mse_loss(predictions, targets)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


def compute_psnr_from_mse(mse):
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()


def evaluate_dataset(model, dataloader):
    model.eval()
    running_metrics = None
    for images_lowres, images_fullres, targets in dataloader:
        images_lowres, images_fullres, targets = utils.device([images_lowres, images_fullres, targets])
        with torch.no_grad():
            predictions = model(images_lowres, images_fullres)
        psnr = compute_psnr(predictions, targets)
        running_metrics = utils.update_metrics(running_metrics, {'psnr': psnr}, len(dataloader))
    model.train()
    return running_metrics


if __name__ == '__main__':
    _, _, test_loader, model, _, _ = conf.params()
    print('Evaluation')
    eval_metrics = evaluate_dataset(model, test_loader)
    print(eval_metrics)
