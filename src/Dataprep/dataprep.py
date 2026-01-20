import pandas as pd
from sklearn.model_selection import StratifiedKFold

class Dataprep:
    def __init__(self,config):
        self.config = config
        self.df_wide = pd.read_csv(self.config.train_path)

    def load_pivot_and_fold(self):
        print(f"Loading the csv file from {self.config.train_path}....")
        try:
            df_long = pd.read_csv(self.config.train_path)
            df_long["image_id"] = df_long["image_path"].apply(lambda x: x.split('/')[-1].split('.')[0])
            df_wide = df_long.pivot(
                
                index = 'image_id',
                columns='target_name',
                values='target'
                ).reset_index()
            meta_df = df_long.drop_duplicates(subset="image_id").drop( columns=["sample_id", "target_name", "target"])
            train_df = meta_df.merge(df_wide,on="image_id",how="left")
            skf = StratifiedKFold(n_splits=self.config.folds, shuffle=True, random_state=self.config.state)
            train_df['fold'] = -1
            for fold, (train_idx, val_idx) in enumerate(skf.split(X=train_df, y=train_df['site'])):
                train_df.loc[val_idx, 'fold'] = fold
            print(f"\nFirst 5 rows:\n{train_df.head()}\n")
            self.train_df = train_df
            return train_df
        except FileNotFoundError:
            print(f"Error: {self.config.train_csv} not found")
            return pd.DataFrame(columns=['image_path'] + self.config.all_target_cols)


           