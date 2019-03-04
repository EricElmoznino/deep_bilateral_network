from torch.nn import functional as F
from tensorboardX import SummaryWriter
import os
import shutil
import sys

import configuration as conf
import utils
from eval import evaluate_dataset, compute_psnr_from_mse

save_dir = os.path.join('saved_runs', sys.argv[1])
shutil.rmtree(save_dir, ignore_errors=True)
os.mkdir(save_dir)
writer = SummaryWriter(os.path.join(save_dir, 'logs'))

n_epochs, train_loader, test_loader, model, optimizer, scheduler = conf.params()

print_freq = 50
best_metrics = None
running_metrics = None
for epoch in range(1, n_epochs + 1):
    print('Starting epoch: %d' % epoch)
    scheduler.step()

    for batch, (images_lowres, images_fullres, targets) in enumerate(train_loader):
        step_num = utils.step_num(epoch, batch, train_loader)

        images_lowres, images_fullres, targets = utils.device([images_lowres, images_fullres, targets])

        predictions = model(images_lowres, images_fullres)
        loss = F.mse_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_metrics = utils.update_metrics(running_metrics,
                                               {'mse': loss.item(),
                                                'psnr': compute_psnr_from_mse(loss)},
                                               print_freq)

        if (step_num + 1) % print_freq == 0:
            utils.log_to_tensorboard(writer, running_metrics, step_num)
            utils.print_metrics(running_metrics, step_num, n_epochs * len(train_loader))
            running_metrics = {l: 0 for l in running_metrics}

    print('Finished epoch %d\n' % epoch)

    print('Evaluation')
    eval_metrics = evaluate_dataset(model, test_loader)
    utils.log_to_tensorboard(writer, eval_metrics, step_num, log_prefix='test')
    utils.print_metrics(eval_metrics, step_num, n_epochs * len(train_loader))

    if best_metrics is None or all([eval_metrics[k] > best_metrics[k] for k in eval_metrics]):
        utils.save_model(model, os.path.join(save_dir, 'model.pth'))
        best_metrics = {k: eval_metrics[k] for k in eval_metrics}

print('Finished Training')
print('Best Validation Metrics:')
print(best_metrics)
writer.close()
