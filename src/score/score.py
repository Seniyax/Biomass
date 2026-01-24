import numpy as np
from sklearn.metrics import r2_score



class scoring_board:
    def __init__(self,r2_weights:list[float]):
        self.r2_weights = r2_weights


    def calculate_score(self,pred_dict:dict[str,np.ndarray],targets_5:np.ndarray) -> float:
        pred_total = pred_dict['total']
        pred_gdm = pred_dict['gdm']
        pred_green = pred_dict['green']

        pred_clover = np.maximum(0,pred_gdm - pred_green)
        pred_dead = np.maximum(0,pred_total - pred_gdm)

        y_preds = np.stack([
            pred_green,pred_dead,pred_clover, pred_gdm, pred_total
        ],axis=1)

        r2_score = r2_score(targets_5,y_preds,multioutput='raw_values')
        weights_score = np.sum(r2_score * self.r2_weights)
        return weights_score
