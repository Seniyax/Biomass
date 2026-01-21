import torch
import numpy as np



def validate_epoch(model,dataloader,criterion,device):
    model.eval()
    loss = 0.0
    all_preds = {'total':[], 'gdm':[], 'green':[]}
    all_target = []

    with torch.no_grad():
        for img_left,img_right,train_targets,all_targets in dataloader:
            img_left = img_left.to(device)
            img_right = img_right.to(device)
            train_targets = train_targets.to(device)

            pred_total,pred_gdm,pred_green = model(img_left,img_right)
            predictions = (pred_total,pred_gdm,pred_green)
            loss = criterion(predictions,train_targets)
            loss += loss.item()

            all_preds['total'].append(pred_total.cpu().numpy())
            all_preds['gdm'].append(pred_gdm.cpu().numpy())
            all_preds['green'].append(pred_green.cpu().numpy())
            all_target.append(all_targets.cpu().numpy())

        preds_np = {
            k:np.concatenate(v).flatten()
            for k,v in all_preds.items()
        }

        targets_np = np.concatenate(all_target)

        
        scores = scorer.calculate_score(preds_np,targets_np)
        avg_loss = loss / len(dataloader)
        return avg_loss, scores