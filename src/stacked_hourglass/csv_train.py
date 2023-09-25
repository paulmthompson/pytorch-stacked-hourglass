import torch
import torch.backends.cudnn
import torch.nn.parallel

import torch.nn as nn

from tqdm import tqdm

from stacked_hourglass.utils.evaluation import accuracy, AverageMeter, final_preds
from stacked_hourglass.utils.transforms import fliplr, flip_back


def do_training_step(model, optimiser, input, target):
    assert model.training, 'model must be in training mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    with torch.enable_grad():
        # Forward pass and loss calculation.
        output = model(input)
        loss = nn.MSELoss()
        loss = sum(loss(o, target) for o in output)
        #loss = sum(diceLoss(o,target).mean() for o in output)

        # Backward pass and parameter update.
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    return output[-1], loss.item()


def do_training_epoch(train_loader, model, device, optimiser, quiet=False):
    losses = AverageMeter()
    f1s = AverageMeter()
    PPVs = AverageMeter()
    sensitivities = AverageMeter()

    # Put the model in training mode.
    model.train()

    iterable = enumerate(train_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Train', total=len(train_loader), ascii=True, leave=True, position=0)
        iterable = progress

    for i, (input, target) in iterable:
        input, target = input.to(device), target.to(device, non_blocking=True)

        output, loss = do_training_step(model, optimiser, input, target)

        f1, PPV, sensitivity = calculate_metrics(output,target)
        
        # measure accuracy and record loss
        losses.update(loss, input.size(0))
        f1s.update(f1,input.size(0))
        PPVs.update(PPV,input.size(0))
        sensitivities.update(sensitivity,input.size(0))
        
        # Show accuracy and loss as part of the progress bar.
        if progress is not None:
            progress.set_description('Loss: {loss:0.4f}, F1: {f1:6.2f}'.format(
                loss=losses.avg,
                f1 = f1s.avg
            ))
            
    return losses.avg, f1s.avg, PPVs.avg, sensitivities.avg


def do_validation_step(model, input, target):
    assert not model.training, 'model must be in evaluation mode.'
    assert len(input) == len(target), 'input and target must contain the same number of examples.'

    # Forward pass and loss calculation.
    output = model(input)
    loss = nn.MSELoss()
    loss = sum(loss(o, target) for o in output)
    #loss = sum(diceLoss(o,target).mean() for o in output)

    heatmaps = output[-1].cpu()

    return heatmaps, loss.item()


def do_validation_epoch(val_loader, model, device, quiet=False):
    losses = AverageMeter()
    f1s = AverageMeter()
    PPVs = AverageMeter()
    sensitivities = AverageMeter()
    
    predictions = [None] * len(val_loader.dataset)

    # Put the model in evaluation mode.
    model.eval()

    iterable = enumerate(val_loader)
    progress = None
    if not quiet:
        progress = tqdm(iterable, desc='Valid', total=len(val_loader), ascii=True, leave=False, position=0)
        iterable = progress

    for i, (input, target) in iterable:
        # Copy data to the training device (eg GPU).
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        heatmaps, loss = do_validation_step(model, input, target)
        
        f1, PPV, sensitivity = calculate_metrics(heatmaps,target)

        # Record accuracy and loss for this batch.
        losses.update(loss, input.size(0))
        f1s.update(f1,input.size(0))
        PPVs.update(PPV,input.size(0))
        sensitivities.update(sensitivity,input.size(0))

        # Show accuracy and loss as part of the progress bar.
        if progress is not None:
            progress.set_postfix_str('Loss: {loss:0.4f}, F1: {f1:6.2f}'.format(
                loss=losses.avg,
                f1 = f1s.avg,
            ))

    #heatmaps.cpu()
    return losses.avg, heatmaps, f1s.avg, PPVs.avg, sensitivities.avg

def calculate_metrics(output,target):
    
    with torch.no_grad():
        output_g = (output > 0.5).to(torch.float32)
    
        tp = (output_g * target).sum()
        fn = ((1 - output_g) * target).sum()
        fp = (output_g * (~ (target == 1.0))).sum()
    
        PPV = tp / ((tp + fp) or 1) #Positive predictive Value
        sensitivity = tp / ((tp + fn) or 1) #Sensitivity
    
        f1 = 2 * (PPV * sensitivity) / ((PPV + sensitivity) or 1)
    
    return f1, PPV, sensitivity

def diceLoss(output,label):
    diceLabel = label.sum()
    dicePrediction = output.sum()
    diceCorrect = (output * label).sum()
    
    diceRatio = (2 * diceCorrect + 1) / (dicePrediction + diceLabel + 1)
    
    return 1 - diceRatio
                                    