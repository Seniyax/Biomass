from src.Dataprep.dataprep import Dataprep
from config.config import CONFIG
from src.Dataset.Dataset import BiomassDataset
from src.model.model import Regressor
from src.preprocessor.Augements import AgumentationFactory
from src.loss.weighted_loss import weightedloss
from src.Trainer.trainer import train_epoch
from src.validation.val import validate_epoch
from src.score.score import scoring_board

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader



def freeze_backbone(model:nn.Module) -> None:
    backbone = model.module.backbone if isinstance(model, nn.DataParallel) else model.backbone
    for param in backbone.parameters():
        param.requires_grad = False

def save_model(model:nn.Module,fold:int) -> None:
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    save_path = config.output_dir / f'best_model_fold{fold}.pth'
    torch.save(state_dict,save_path)

def main():
   config = CONFIG()
   config.display_info()
   data = Dataprep(config)
   data.load_pivot_and_fold()
   augfac = AgumentationFactory(config.img_size)
   scorer = scoring_board(config.r2_weights)

   for fold in range(config.folds):
       print(f"\n========== FOLD {fold} ==========")
       train_fold_df = df[df.fold != fold].reset_index(drop=True)
       val_fold_indx = df[df.fold == fold]
       val_indices = val_fold_indx.index
       val_fold_df = val_fold_indx.reset_index(drop=True)

       train_dataset = BiomassDataset(
        train_fold_df,
        config.train_img,
        augfac.train_transform(),
        config.train_target_cols,
        config.all_target_cols
    )

       vald_dataset = BiomassDataset(
        val_fold_df,
        config.train_img,
        augfac.valid_transform(),
        config.train_target_cols,
        config.all_target_cols
    )

       train_loader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,num_workers=os.cpu_count())
       valid_loader = DataLoader(vald_dataset,batch_size=config.batch_size * 2,shuffle=False,num_workers=os.cpu_count())

       model_base = Regressor(config.model,config.pretrained)
       model_base.to(config.device)

       print(f"Freezing the Backbone of {config.model}")
       freeze_backbone(model_base)

       optimizer = optim.AdamW(filter(lambda p:p.requires_grad,model_base.parameters()),lr=config.train_lr,weight_decay=1e-3)
       criterion = weightedLoss(config.loss_weights).to(config.device)
       scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = config.epoches,eta_min = 1e-6)

       best_score = -float('inf')

       for epoch in range(config.epoches):
           
           train_loss = train_epoch(model_base,train_loader,criterion,optimizer,scheduler,config.device)
           val_loss,scores = validate_epoch(model_base,valid_loader,criterion,config.device)

           scheduler.step()

           print(f"epoch:{epoch} | Training loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | wR²: {scores:.4f}")

           #if scores > best_score:
                #best_score = scores
                #save_model(model_base, fold)
                #print(" R² score improved! Saving model")

if __name__ = '__main__':
    main()
    print("="*70) 
   
