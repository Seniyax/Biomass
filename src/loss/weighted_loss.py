import torch
import torch.nn as nn


class weightedloss:
    def __init__(self,loss_weights:dict[str,float]):
        self.loss_weights = loss_weights
        self.criterion = nn.SmoothL1Loss()

    def forward(self,predictions: tuple[torch.Tensor,torch.Tensor,torch.Tensor],targets:torch.Tensor) -> torch.tensor:

        pred_total,pred_gdm,pred_green = predictions

        loss_total = self.criterion(pred_total,targets[:, 0:1])
        loss_gdm = self.criterion(pred_gdm,targets[:, 1:2])
        loss_green = self.criterion(pred_green,targets[:, 2:3])

        total_loss = (
            self.weights['total_loss'] * loss_total +
            self.weights['gdm_loss'] * loss_gdm +
            self.weights['green_loss'] * loss_green
        )
        
        return total_loss

        