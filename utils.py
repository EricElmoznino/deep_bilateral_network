import torch


def device(tensor_list):
    if torch.cuda.is_available():
        return [t.cuda() for t in tensor_list]
    else:
        return tensor_list


def step_num(epoch, batch, dataloader):
    return (epoch - 1) * len(dataloader) + batch


def update_metrics(running_losses, losses, scale):
    if running_losses is None:
        running_losses = {l: 0 for l in losses}
    for l in losses:
        running_losses[l] += losses[l] / scale
    return running_losses


def print_metrics(running_losses, step, n_steps):
    print('Step [%d / %d]' % (step, n_steps))
    for loss_name, loss in running_losses.items():
        print('%s: %.5f' % (loss_name, loss))
    print('')


def log_to_tensorboard(writer, running_losses, step, log_prefix='training'):
    location = log_prefix + '/'
    for loss_name, loss in running_losses.items():
        writer.add_scalar(location + loss_name, loss, step)


def save_model(model, save_path):
    model.eval()
    torch.save(model.state_dict(), save_path)
    model.train()


def load_model(model, save_path, strict=True):
    model.eval()
    model.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage), strict=strict)
    model.train()
