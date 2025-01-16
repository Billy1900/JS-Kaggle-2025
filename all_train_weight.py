import torch
import torch.nn as nn
import torch.optim

import numpy as np
from tqdm import tqdm
import polars as pl
import argparse
import pickle
import os
from tanm_reference import Model, make_parameter_groups


from pytorch_lightning import LightningModule


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define R2 loss class
# class R2Loss(nn.Module):
#     def __init__(self):
#         super(R2Loss, self).__init__()

#     def forward(self, y_pred, y_true):
#         mse_loss = torch.sum((y_pred - y_true) ** 2)
#         var_y = torch.sum(y_true ** 2)
#         loss = mse_loss / (var_y + 1e-38)
#         return loss


class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true, sample_weight):
        # Calculate weighted residual sum of squares (numerator)
        residuals = sample_weight * (y_true - y_pred) ** 2
        weighted_residual_sum = torch.sum(residuals)

        # Calculate weighted sum of squared true values (denominator)
        weighted_true_sum = torch.sum(sample_weight * (y_true) ** 2)

        # Compute weighted R2 loss
        loss = weighted_residual_sum / (weighted_true_sum + 1e-38)
        return loss

def r2_val(y_true, y_pred, sample_weight):
    residuals = sample_weight * (y_true - y_pred) ** 2
    weighted_residual_sum = np.sum(residuals)
    weighted_true_sum = np.sum(sample_weight * (y_true) ** 2)
    return 1 - weighted_residual_sum / weighted_true_sum

# Neural Network class using PyTorch Lightning
class NN(LightningModule):
    def __init__(self, n_cont_features, cat_cardinalities, n_classes, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.k = 32
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.loss_fn = R2Loss()

        self.model = Model(
            n_num_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes,
            backbone={
                'type': 'MLP',
                'n_blocks': 3,
                'd_block': 512,
                'dropout': 0.25,
            },
            bins=None,
            num_embeddings=None,
            arch_type='tabm',
            k=self.k,
        )

    def forward(self, x_cont, x_cat):
        return self.model(x_cont, x_cat).squeeze(-1)


def train_model(
    model,
    optimizer,
    loss_fn,
    batch_tensors,       # 按天分好的数据 (dict: date_id -> Tensor)
    train_tensor_all,    # 整个训练集 (Tensor)
    valid_tensor_all,    # 整个验证集 (Tensor)
    unique_dates,
    num_epochs,
    small_batch_size,
    save_dir
):
    k = 32

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # ------------------------------------------------------
        # 1) 在 epoch 开始时，对 train_tensor_all 做一次shuffle
        # ------------------------------------------------------
        print("shuffle train_data")
        shuffle_indices = torch.randperm(train_tensor_all.shape[0], device=train_tensor_all.device)
        # 建议：如果 train_tensor_all 很大，且显存不够，也可以在CPU上做 shuffle，然后再 .to(device)
        train_tensor_all_shuffled = train_tensor_all[shuffle_indices]
        print("end shuffle")

        # 指针从 0 开始，后面依次切 2n 行
        pointer = 0

        model.train()
        total_loss = 0.0
        total_count = 0
        num_sub_batches_total = 0

        # 为了计算 epoch 级别的训练 R2，需要记录预测、标签和权重
        train_pred_list = []

        with tqdm(unique_dates, desc=f"Epoch {epoch + 1}", unit="batch") as progress_bar:
            for date in progress_bar:
                # 当前天的数据
                batch_tensor = batch_tensors[date]
                n = batch_tensor.shape[0]

                # ------------------------------------------------
                # 2) 从洗好的 train_tensor_all_shuffled 里顺序取 2n 行
                # ------------------------------------------------
                # 如果即将越界，直接把 pointer 重置为 0
                if pointer + n > train_tensor_all_shuffled.shape[0]:
                    pointer = 0

                # 切片取 n 行
                random_samples = train_tensor_all_shuffled[pointer : pointer + n]
                pointer += n

                # 3) 拼接成 2n 行，然后 shuffle
                combined_data = torch.cat([batch_tensor, random_samples], dim=0)
                shuffle_indices_local = torch.randperm(combined_data.shape[0])
                combined_data = combined_data[shuffle_indices_local].to(device)

                # 4) 细分成小批 sub-batch（同你原先的逻辑）
                total_rows = combined_data.shape[0]
                num_sub_batches = (total_rows + small_batch_size - 1) // small_batch_size
                num_sub_batches_total += num_sub_batches

                for sub_batch_idx in range(num_sub_batches):
                    start_idx = sub_batch_idx * small_batch_size
                    end_idx = min(start_idx + small_batch_size, total_rows)
                    sub_batch_tensor = combined_data[start_idx:end_idx]

                    # 根据你的列切分
                    X_input = sub_batch_tensor[:, :-4]
                    y_input = sub_batch_tensor[:, -4]
                    w_input = sub_batch_tensor[:, -5]    
                    symbol_input = sub_batch_tensor[:, -3]
                    time_input = sub_batch_tensor[:, -2]

                    # data augmentation
                    x_cont_input = X_input + torch.randn_like(X_input) * 0.2
                    x_cat_input = torch.concat(
                        [symbol_input.unsqueeze(-1), time_input.unsqueeze(-1)], axis=1
                    ).to(torch.int64)

                    # forward & backward
                    optimizer.zero_grad()
                    output = model(x_cont_input, x_cat_input).squeeze(-1)
                    loss = loss_fn(
                        output.flatten(0, 1), 
                        y_input.repeat_interleave(k), 
                        w_input.repeat_interleave(k)  # 添加权重
                    )
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_count += 1

                    # 用于计算全 epoch 训练 R2
                    train_pred_list.append((
                        output.mean(dim=1).detach().cpu(),
                        y_input.cpu(),
                        w_input.cpu()
                    ))

                    progress_bar.set_postfix({
                        "Date": date,
                        "Sub-batch": f"{sub_batch_idx + 1}/{num_sub_batches}",
                        "Loss": f"{loss.item():.6f}",
                        "Avg_loss": f"{total_loss/total_count:.6f}",
                    })

                # 及时释放显存
                del batch_tensor, combined_data, random_samples
                torch.cuda.empty_cache()

        # 计算本轮 epoch 的平均 train loss
        avg_train_loss = total_loss / num_sub_batches_total
        
        # 计算 train R2
        pred_all = torch.cat([x[0] for x in train_pred_list], dim=0).numpy()
        y_all = torch.cat([x[1] for x in train_pred_list], dim=0).numpy()
        w_all = torch.cat([x[2] for x in train_pred_list], dim=0).numpy()
        train_r2 = r2_val(y_all, pred_all, w_all)

        # 做一次验证
        valid_loss, valid_r2 = validate(model, loss_fn, valid_tensor_all, small_batch_size, k)
        
        print(f"[Epoch {epoch+1}] "
              f"Train Loss={avg_train_loss:.6f}, Train R2={train_r2:.6f}, "
              f"Valid Loss={valid_loss:.6f}, Valid R2={valid_r2:.6f}"
        )
        
        # 保存 checkpoint（同你原先的逻辑）
        checkpoint_filename = (
            f"retrain_epoch{epoch+1+2}_"
            f"trainloss_{avg_train_loss:.6f}_"
            f"trainr2_{train_r2:.6f}_"
            f"validloss_{valid_loss:.6f}_"
            f"validr2_{valid_r2:.6f}.ckpt"
        )
        checkpoint_path = os.path.join(save_dir, checkpoint_filename)
        import pytorch_lightning as pyl
        checkpoint = {
            'pytorch-lightning_version': pyl.__version__,
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyper_parameters': model.hparams if hasattr(model, 'hparams') else {},
            'metrics': {
                'train_loss': avg_train_loss,
                'train_r2': train_r2,
                'valid_loss': valid_loss,
                'valid_r2': valid_r2,
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}\n")


@torch.no_grad()
def validate(model, loss_fn, valid_tensor_all, small_batch_size, k):
    """
    对整个验证集做一次验证，返回 (valid_loss, valid_r2)
    """
    model.eval()
    valid_size = valid_tensor_all.shape[0]
    num_sub_batches = (valid_size + small_batch_size - 1) // small_batch_size
    
    valid_losses = []
    pred_list = []
    
    for sub_batch_idx in tqdm(range(num_sub_batches)):
        start_idx = sub_batch_idx * small_batch_size
        end_idx = min(start_idx + small_batch_size, valid_size)
        sub_batch = valid_tensor_all[start_idx:end_idx].to(device)

        X_valid = sub_batch[:, :-4]
        y_valid = sub_batch[:, -4]
        w_valid = sub_batch[:, -5]
        symbol_valid = sub_batch[:, -3]
        time_valid = sub_batch[:, -2]
        
        x_cont_valid = X_valid + torch.randn_like(X_valid) * 0.2
        x_cat_valid = torch.concat(
            [symbol_valid.unsqueeze(-1), time_valid.unsqueeze(-1)], axis=1
        ).to(torch.int64)

        y_pred = model(x_cont_valid, x_cat_valid).squeeze(-1)
        
        # 计算损失
        val_loss = loss_fn(
            y_pred.flatten(0, 1), 
            y_valid.repeat_interleave(k),
            w_valid.repeat_interleave(k)  # 添加权重
        )
        valid_losses.append(val_loss.item())

        # 收集预测，用于后面计算 R2
        pred_list.append((
            y_pred.mean(dim=1).cpu(),  # (batch, k) -> (batch,)
            y_valid.cpu(),
            w_valid.cpu()
        ))

    avg_valid_loss = sum(valid_losses) / len(valid_losses)
    
    # 计算 R2
    pred_all = torch.cat([p[0] for p in pred_list], dim=0).numpy()
    y_all = torch.cat([p[1] for p in pred_list], dim=0).numpy()
    w_all = torch.cat([p[2] for p in pred_list], dim=0).numpy()
    valid_r2 = r2_val(y_all, pred_all, w_all)

    return avg_valid_loss, valid_r2



def prepare_data(saved=False):
    base_path = "/mnt/sda/k/"

    if saved:
        train_original = pl.read_parquet(f"{base_path}train_original.parquet")
        valid_original = pl.read_parquet(f"{base_path}valid_original.parquet")

        return train_original,valid_original

    train_df = pl.scan_parquet(f"{base_path}train.parquet")

    lag_df = (
        train_df
        .select(
            [
                pl.col("date_id"),
                pl.col("symbol_id"),
                pl.col("time_id"),
            ]
            + [pl.col(f"responder_{i}") for i in range(9)]
        )
        # 把 date_id -> date_id + 1
        .with_columns(
            (pl.col("date_id") + 1).alias("date_id_next")
        )
        # 重命名 responder_i -> responder_i_lag_1
        .rename({f"responder_{i}": f"responder_{i}_lag_1" for i in range(9)})
    )

    lag_data = train_df.join(
        lag_df,
        left_on=["date_id", "symbol_id", "time_id"],
        right_on=["date_id_next", "symbol_id", "time_id"],
        how="left", 
    )

    lag_data = lag_data.drop("date_id_right").fill_nan(0).fill_null(0)

    print("lag_data.collect() ... ")

    lag_data = lag_data.collect()

    feature_train_list = [f"feature_{idx:02d}" for idx in range(79) ] 

    target_col = "responder_6"

    feature_train = feature_train_list + [f"responder_{idx}_lag_1" for idx in range(9)] 

    start_dt = 252
    end_dt = 1577


    # train_original = lag_data.filter((pl.col("date_id") >=start_dt) & (pl.col("date_id") < end_dt))
    train_original = lag_data.filter((pl.col("date_id") >=start_dt))

    valid_original = lag_data.filter(pl.col("date_id") > end_dt)


    category_mappings = {
        'symbol_id' : {i : i for i in range(60)},
        'time_id' : {i : i for i in range(1000)}
    }

    def encode_column(df, column, mapping):
        def encode_category(category):
            return mapping.get(category, -1)  
        
        return df.with_columns(
            pl.col(column).map_elements(encode_category, return_dtype=pl.Int16).alias(column)
        )

    print("encode")

    for col in ['symbol_id', 'time_id']:
        train_original = encode_column(train_original, col, category_mappings[col])
        valid_original = encode_column(valid_original, col, category_mappings[col])



    import json
    with open("data_stats.json", "r") as file:
        data_stats = json.load(file)

    means = data_stats['means']
    stds = data_stats['stds']

    def standardize(df, feature_cols, means, stds):
        return df.with_columns([
            ((pl.col(col) - means[col]) / stds[col]).alias(col) for col in feature_cols
        ])

    print("standarlize")
    train_original = standardize(train_original,feature_train,means,stds)
    valid_original = standardize(valid_original,feature_train,means,stds)

    # train_original = train_original.select(feature_train + [target_col, 'weight', 'symbol_id', 'time_id', 'date_id'])
    # valid_original = valid_original.select(feature_train + [target_col, 'weight', 'symbol_id', 'time_id', 'date_id'])

    train_original = train_original.select(feature_train + ['weight', target_col, 'symbol_id', 'time_id', 'date_id'])
    valid_original = valid_original.select(feature_train + ['weight', target_col, 'symbol_id', 'time_id', 'date_id'])


    # print("write train & valid data")

    # train_original.write_parquet(f"{base_path}train_original.parquet")
    # valid_original.write_parquet(f"{base_path}valid_original.parquet")

    return train_original,valid_original

def get_batch_tensors(saved=False,all_data=False):
    base_path = "/mnt/sda/k/"

    if saved:
        if all_data:
            print("load all data.")
            with open(f"{base_path}w_all_data_train_batch_tensors.pkl", "rb") as f:
                train_batch_tensors = pickle.load(f)
            with open(f"{base_path}w_all_data_train_tensor_all.pkl", "rb") as f:
                train_tensor_all = pickle.load(f)
            with open(f"{base_path}w_valid_tensor_all.pkl", "rb") as f:
                valid_tensor_all = pickle.load(f)
        else:
            print("load train data")
            with open(f"{base_path}w_train_batch_tensors.pkl", "rb") as f:
                train_batch_tensors = pickle.load(f)
            with open(f"{base_path}w_train_tensor_all.pkl", "rb") as f:
                train_tensor_all = pickle.load(f)
            with open(f"{base_path}w_valid_tensor_all.pkl", "rb") as f:
                valid_tensor_all = pickle.load(f)

        
        return train_batch_tensors, train_tensor_all, valid_tensor_all

    # 原先的 prepare_data 会返回两个 polars.DataFrame
    train_original, valid_original = prepare_data(saved=False)

    # 1) 整个训练集 Tensor
    train_tensor_all = torch.tensor(
        train_original.to_numpy(), dtype=torch.float32
    )

    # 2) 将训练集按天拆分的字典（你已有的逻辑）
    train_batch_tensors = {}
    unique_dates = train_original["date_id"].unique().to_list()
    for date in tqdm(unique_dates):
        batch_data = train_original.filter(pl.col("date_id") == date)
        train_batch_tensors[date] = torch.tensor(
            batch_data.to_numpy(), dtype=torch.float32
        )
    
    # 3) 整个验证集 Tensor（方便做验证循环）
    valid_tensor_all = torch.tensor(
        valid_original.to_numpy(), dtype=torch.float32
    )    # 这里也可以不 to(device)，留到后面验证时再转
    
    # 4) 保存 train_batch_tensors 到文件（如有需要）
    with open(f"{base_path}w_all_data_train_batch_tensors.pkl", "wb") as f:
        pickle.dump(train_batch_tensors, f)
    
    with open(f"{base_path}w_all_data_train_tensor_all.pkl", "wb") as f:
        pickle.dump(train_tensor_all, f)

    with open(f"{base_path}w_valid_tensor_all.pkl", "wb") as f:
        pickle.dump(valid_tensor_all, f)

    return train_batch_tensors, train_tensor_all, valid_tensor_all


    

class Custom_Args():
    def __init__(self):
        self.usegpu = True
        self.gpuid = 0
        self.seed = 42
        self.model = 'nn'
        self.use_wandb = False
        self.project = 'js-tabm-with-lags'
        self.dname = "./input_df/"
        self.loader_workers = 10   
        self.bs = 8192
        self.lr = 1e-3
        self.weight_decay = 8e-4
        self.n_cont_features = 89
        self.n_cat_features = 2
        self.n_classes = None
        self.cat_cardinalities = [60, 1000]
        self.patience = 7
        self.max_epochs = 10
        self.N_fold = 5




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load model checkpoint')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--small_batch_size', type=int, default=8192, help='Small batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--all_data', type=int, default=0, help='Use all data or not')
    args = parser.parse_args()

    print(args)

    # 拿到3个东西：分天的字典、完整训练集tensor、完整验证集tensor
    print("get 3 tensors ...")
    train_batch_tensors, train_tensor_all, valid_tensor_all = get_batch_tensors(saved=True,all_data=args.all_data)
    print("completed.")
    unique_dates = sorted(list(train_batch_tensors.keys()))

    if args.load_path:
        print("load model:",args.load_path)
        model = NN.load_from_checkpoint(args.load_path).to(device)
    else:
        print("init model")
        my_args = Custom_Args()
        model = NN(
            my_args.n_cont_features,
            my_args.cat_cardinalities,
            my_args.n_classes,
            my_args.lr,
            my_args.weight_decay
        ).to(device)

    optimizer = torch.optim.AdamW(
        make_parameter_groups(model),
        lr=args.lr,
        weight_decay=7e-3,
    )
    loss_fn = R2Loss()

    os.makedirs(args.save_dir, exist_ok=True)

    train_model(
        model,
        optimizer,
        loss_fn,
        batch_tensors=train_batch_tensors,
        train_tensor_all=train_tensor_all,
        valid_tensor_all=valid_tensor_all,
        unique_dates=unique_dates,
        num_epochs=args.num_epochs,
        small_batch_size=args.small_batch_size,
        save_dir=args.save_dir
    )
