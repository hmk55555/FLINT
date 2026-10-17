import os
import json
import torch

# from colbert.utils.runs import Run
from colbert.utils.utils import print_message, save_checkpoint
from colbert.parameters import SAVED_CHECKPOINTS
from colbert.infra.run import Run


def print_progress(scores):
    positive_avg, negative_avg = round(scores[:, 0].mean().item(), 2), round(scores[:, 1].mean().item(), 2)
    print("#>>>   ", positive_avg, negative_avg, '\t\t|\t\t', positive_avg - negative_avg)


def manage_checkpoints(args, colbert, optimizer, batch_idx, savepath=None, consumed_all_triples=False, scheduler=None):
    # arguments = dict(args)

    # TODO: Call provenance() on the values that support it??

    checkpoints_path = savepath or os.path.join(Run().path_, 'checkpoints')
    name = None

    try:
        save = colbert.save
    except:
        save = colbert.module.save

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    
    path_save = None

    if consumed_all_triples or (batch_idx % 2000 == 0):
        # name = os.path.join(path, "colbert.dnn")
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, "colbert")

    if batch_idx in SAVED_CHECKPOINTS:
        # name = os.path.join(path, "colbert-{}.dnn".format(batch_idx))
        # save_checkpoint(name, 0, batch_idx, colbert, optimizer, arguments)
        path_save = os.path.join(checkpoints_path, f"colbert-{batch_idx}")

    if path_save:
        print(f"#> Saving a checkpoint to {path_save} ..")

        checkpoint = {}
        checkpoint['batch'] = batch_idx
        # checkpoint['epoch'] = 0
        # checkpoint['model_state_dict'] = model.state_dict()
        # checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        # checkpoint['arguments'] = arguments

        save(path_save)

        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(checkpoint, os.path.join(path_save, 'optimizer_state.pt'))

        with open(os.path.join(path_save, 'training_state.json'), 'w') as f:
            json.dump({'step': batch_idx, 'maxsteps': args.maxsteps, 'stop_at_step': args.stop_at_step,
                       'finished': bool(consumed_all_triples)}, f, indent=2)

    return path_save


def load_resume_state(checkpoint_path):
    return torch.load(os.path.join(checkpoint_path, 'optimizer_state.pt'), map_location='cpu')
