import torch
import torch.nn as nn
import torch.optim

from pytorch_lightning import LightningModule

import numpy as np
from tqdm import tqdm
import polars as pl
import argparse
import pickle
import os

from tanm_reference import Model, make_parameter_groups

# Device configuration - use GPU if available, otherwise CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define R2 (R-squared) loss class for regression tasks
class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.sum((y_pred - y_true) ** 2)
        var_y = torch.sum(y_true ** 2)
        loss = mse_loss / (var_y + 1e-38)  # Add small epsilon to prevent division by zero
        return loss

# Calculate R2 score for validation
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
        self.lr = lr
        self.weight_decay = weight_decay

        # Create 10 parallel models for processing sequential data
        # Each model processes one time step in the sequence
        self.models = nn.ModuleList([
            Model(
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
                k=2,  # Reduced from k=32 as this is a different approach
            )
            for _ in range(10)
        ])
        
        # Linear layer to combine outputs from the 10 parallel models
        # Input: (batch_size, 20), Output: (batch_size, 1)
        self.final_fc = nn.Linear(20, 1)
        self.loss_fn = R2Loss()

    def forward(self, x_cont, x_cat):
        """
        Forward pass through the network
        Args:
            x_cont: Continuous features (batch_size, window_size, n_cont_features)
            x_cat: Categorical features (batch_size, window_size, n_cat_features)
        Returns:
            output: Predictions (batch_size,)
        """
        # Collect outputs from each time step
        outputs = []
        for i in range(x_cont.size(1)):  # Iterate over window_size dimension
            # Get data for current time step
            x_cont_i = x_cont[:, i, :]  # (batch_size, n_cont_features)
            x_cat_i = x_cat[:, i, :]    # (batch_size, n_cat_features)
            
            # Process through corresponding sub-model
            out_i = self.models[i](x_cont_i, x_cat_i)  # (batch_size, 1)
            out_i = out_i.squeeze(-1)  # Remove last dimension
            outputs.append(out_i)
        
        # Concatenate all time step outputs: (batch_size, window_size)
        merged = torch.cat(outputs, dim=1)
        # Final prediction through fully connected layer: (batch_size, 1)
        final_out = self.final_fc(merged)
        
        return final_out.squeeze(-1)  # (batch_size,)

def get_windows_from_padded(padded_data, window_size=10):
    """
    Extract time windows from padded data
    Args:
        padded_data: Padded input data
        window_size: Size of the sliding window
    Returns:
        x_cont: Continuous features
        x_cat: Categorical features
        y: Target values
        w: Sample weights
    """
    # Calculate length of data for each symbol (including padding)
    symbol_lengths = []
    current_symbol = padded_data[0, -3]
    current_length = 0
    
    for i in range(len(padded_data)):
        if padded_data[i, -3] != current_symbol:
            symbol_lengths.append(current_length)
            current_symbol = padded_data[i, -3]
            current_length = 1
        else:
            current_length += 1
    symbol_lengths.append(current_length)
    
    # Prepare containers for windows, targets, and weights
    windows = []
    targets = []
    weights = []
    
    start_idx = 0
    for length in symbol_lengths:
        symbol_data = padded_data[start_idx:start_idx+length]
        
        # Create windows for each valid position
        for i in range(window_size-1, length):
            window = symbol_data[i-window_size+1:i+1]
            target = symbol_data[i, -5]  # responder_6
            weight = symbol_data[i, -4]  # weight
            
            windows.append(window)
            targets.append(target)
            weights.append(weight)
        
        start_idx += length
    
    # Stack and prepare final tensors
    windows = torch.stack(windows)
    x_cont = windows[:, :, :-5]
    x_cat = windows[:, :, [-3, -2]]
    y = torch.stack(targets)
    w = torch.stack(weights)
    
    return x_cont, x_cat.to(torch.int64), y, w

def train_model(
    model,
    optimizer,
    loss_fn,
    batch_tensors,  # Data split by days (dict: date_id -> Tensor)
    train_tensor_all,       
    valid_tensor_all,    # Complete validation set (Tensor)
    unique_dates,
    num_epochs,
    small_batch_size=256,
    window_size=10,
    save_dir='checkpoints'
):
    """
    Train the model sequentially by processing data day by day
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        model.train()
        total_loss = 0.0
        batch_count = 0
        
        # Collectors for training predictions and labels
        train_pred_list = []
        
        with tqdm(unique_dates, desc=f"Epoch {epoch + 1}", unit="batch") as progress_bar:
            for date in progress_bar:
                # Get data for current day and convert to time windows
                batch_tensor = batch_tensors[date]
                x_cont, x_cat, y, w = get_windows_from_padded(batch_tensor, window_size)

                x_cont  = x_cont.to(device)
                x_cat   = x_cat.to(device)
                y       = y.to(device)
                w       = w.to(device)

                if len(y) == 0:  # Skip empty batches
                    continue
                
                # Calculate number of mini-batches needed
                num_samples = len(y)
                num_batches = (num_samples + small_batch_size - 1) // small_batch_size
                
                # Iterate over mini-batches
                for i in range(num_batches):
                    start_idx = i * small_batch_size
                    end_idx = min(start_idx + small_batch_size, num_samples)
                    
                    # Get current mini-batch
                    x_cont_batch = x_cont[start_idx:end_idx]
                    x_cat_batch = x_cat[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx]
                    w_batch = w[start_idx:end_idx]
                    
                    # Forward pass and training
                    optimizer.zero_grad()
                    output = model(x_cont_batch, x_cat_batch)
                    loss = loss_fn(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Collect predictions
                    train_pred_list.append((
                        output.detach().cpu(),
                        y_batch.cpu(),
                        w_batch.cpu()
                    ))
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "Date": date,
                        "Sub-batch": f"{i + 1}/{num_batches}",
                        "Loss": f"{loss.item():.6f}",
                        "Avg_loss": f"{total_loss/batch_count:.6f}",
                    })
                
                # Free GPU memory
                del x_cont, x_cat, y, w, batch_tensor
                torch.cuda.empty_cache()
        
        # Calculate training metrics
        avg_train_loss = total_loss / batch_count
        pred_all = torch.cat([x[0] for x in train_pred_list], dim=0).numpy()
        y_all = torch.cat([x[1] for x in train_pred_list], dim=0).numpy()
        w_all = torch.cat([x[2] for x in train_pred_list], dim=0).numpy()
        train_r2 = r2_val(y_all, pred_all, w_all)
        
        # Validate
        valid_loss, valid_r2 = validate(
            model, loss_fn, valid_tensor_all, window_size, small_batch_size
        )
        
        # Print epoch results
        print(f"[Epoch {epoch+1}] "
              f"Train Loss={avg_train_loss:.6f}, Train R2={train_r2:.6f}, "
              f"Valid Loss={valid_loss:.6f}, Valid R2={valid_r2:.6f}")
        
        # Save checkpoint
        checkpoint_filename = (
            f"timewindow_epoch{epoch+1}_"
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
def validate(model, loss_fn, valid_tensor_all, window_size, small_batch_size):
    """
    Validate model performance day by day, supporting mini-batch processing
    """
    model.eval()
    valid_losses = []
    pred_list = []
    
    for date, batch_tensor in valid_tensor_all.items():
        # Get data for current day
        x_cont, x_cat, y, w = get_windows_from_padded(batch_tensor, window_size)
        
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        w = w.to(device)
        
        if len(y) == 0:  # Skip empty batches
            continue
            
        # Calculate number of mini-batches needed
        num_samples = len(y)
        num_batches = (num_samples + small_batch_size - 1) // small_batch_size
        
        # Iterate over mini-batches
        for i in range(num_batches):
            start_idx = i * small_batch_size
            end_idx = min(start_idx + small_batch_size, num_samples)
            
            # Get current mini-batch
            x_cont_batch = x_cont[start_idx:end_idx]
            x_cat_batch = x_cat[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            w_batch = w[start_idx:end_idx]
            
            # Make predictions
            output = model(x_cont_batch, x_cat_batch)
            val_loss = loss_fn(output, y_batch)
            valid_losses.append(val_loss.item())
            
            # Collect predictions
            pred_list.append((
                output.cpu(),
                y_batch.cpu(),
                w_batch.cpu()
            ))
        
        # Free GPU memory
        del x_cont, x_cat, y, w, batch_tensor
        torch.cuda.empty_cache()
    
    # Calculate validation metrics
    avg_valid_loss = sum(valid_losses) / len(valid_losses)
    pred_all = torch.cat([p[0] for p in pred_list], dim=0).numpy()
    y_all = torch.cat([p[1] for p in pred_list], dim=0).numpy()
    w_all = torch.cat([p[2] for p in pred_list], dim=0).numpy()
    valid_r2 = r2_val(y_all, pred_all, w_all)
    
    return avg_valid_loss, valid_r2

def prepare_data(saved=False):
    """
    Prepare and preprocess training and validation data
    Args:
        saved: Whether to load previously saved data
    Returns:
        train_original: Training dataset
        valid_original: Validation dataset
    """
    base_path = "/mnt/sda/k/"

    if saved:
        train_original = pl.read_parquet(f"{base_path}train_original.parquet")
        valid_original = pl.read_parquet(f"{base_path}valid_original.parquet")
        return train_original, valid_original

    # Load and process training data
    train_df = pl.scan_parquet(f"{base_path}train.parquet")

    # Create lag features
    lag_df = (
        train_df
        .select(
            [
                pl.col("time_id"),
            ]
            + [pl.col(f"responder_{i}") for i in range(9)]
        )
        # Shift date_id forward by 1
        .with_columns(
            (pl.col("date_id") + 1).alias("date_id_next")
        )
        # Rename responder columns to include lag indicator
        .rename({f"responder_{i}": f"responder_{i}_lag_1" for i in range(9)})
    )

    # Join lagged data with original data
    lag_data = train_df.join(
        lag_df,
        left_on=["date_id", "symbol_id", "time_id"],
        right_on=["date_id_next", "symbol_id", "time_id"],
        how="left", 
    )

    lag_data = lag_data.drop("date_id_right").fill_nan(0).fill_null(0)

    print("lag_data.collect() ... ")
    lag_data = lag_data.collect()

    # Define feature lists and target column
    feature_train_list = [f"feature_{idx:02d}" for idx in range(79)]
    target_col = "responder_6"
    feature_train = feature_train_list + [f"responder_{idx}_lag_1" for idx in range(9)]

    # Define date ranges for train/validation split
    start_dt = 252
    end_dt = 1577

    # Split data into train and validation sets
    train_original = lag_data.filter((pl.col("date_id") >= start_dt) & (pl.col("date_id") < end_dt))
    valid_original = lag_data.filter(pl.col("date_id") > end_dt)

    # Create category mappings for categorical variables
    category_mappings = {
        'symbol_id': {i: i for i in range(60)},
        'time_id': {i: i for i in range(1000)}
    }

    def encode_column(df, column, mapping):
        """Encode categorical columns using provided mapping"""
        def encode_category(category):
            return mapping.get(category, -1)
        
        return df.with_columns(
            pl.col(column).map_elements(encode_category, return_dtype=pl.Int16).alias(column)
        )

    print("encode")
    # Encode categorical columns
    for col in ['symbol_id', 'time_id']:
        train_original = encode_column(train_original, col, category_mappings[col])
        valid_original = encode_column(valid_original, col, category_mappings[col])

    # Load data statistics for standardization
    import json
    with open("data_stats.json", "r") as file:
        data_stats = json.load(file)

    means = data_stats['means']
    stds = data_stats['stds']

    def standardize(df, feature_cols, means, stds):
        """Standardize numerical features"""
        return df.with_columns([
            ((pl.col(col) - means[col]) / stds[col]).alias(col) for col in feature_cols
        ])

    print("standardize")
    train_original = standardize(train_original, feature_train, means, stds)
    valid_original = standardize(valid_original, feature_train, means, stds)

    # Select final columns
    train_original = train_original.select(feature_train + [target_col, 'weight', 'symbol_id', 'time_id', 'date_id'])
    valid_original = valid_original.select(feature_train + [target_col, 'weight', 'symbol_id', 'time_id', 'date_id'])

    return train_original, valid_original

def get_batch_tensors(saved=False, all_data=False):
    """
    Get tensors for batch processing
    Args:
        saved: Whether to load previously saved tensors
        all_data: Whether to use all data or not
    Returns:
        train_batch_tensors: Dictionary of training tensors by date
        train_tensor_all: Complete training tensor
        valid_tensor_all: Complete validation tensor
    """
    base_path = "/mnt/sda/k/"

    if saved:
        if all_data:
            print("load all data.")
            with open(f"{base_path}all_data_train_batch_tensors.pkl", "rb") as f:
                train_batch_tensors = pickle.load(f)
            with open(f"{base_path}all_data_train_tensor_all.pkl", "rb") as f:
                train_tensor_all = pickle.load(f)
            with open(f"{base_path}valid_tensor_all.pkl", "rb") as f:
                valid_tensor_all = pickle.load(f)
        else:
            print("load train data")
            with open(f"{base_path}train_batch_tensors.pkl", "rb") as f:
                train_batch_tensors = pickle.load(f)
            with open(f"{base_path}train_tensor_all.pkl", "rb") as f:
                train_tensor_all = None
            with open(f"{base_path}valid_tensor_all.pkl", "rb") as f:
                valid_tensor_all = pickle.load(f)

        return train_batch_tensors, train_tensor_all, valid_tensor_all

    # Get preprocessed data
    train_original, valid_original = prepare_data(saved=False)

    # Create complete training tensor
    train_tensor_all = torch.tensor(
        train_original.to_numpy(), dtype=torch.float32
    )

    # Split training data by date
    train_batch_tensors = {}
    unique_dates = train_original["date_id"].unique().to_list()
    for date in tqdm(unique_dates):
        batch_data = train_original.filter(pl.col("date_id") == date)
        train_batch_tensors[date] = torch.tensor(
            batch_data.to_numpy(), dtype=torch.float32
        )
    
    # Create complete validation tensor
    valid_tensor_all = torch.tensor(
        valid_original.to_numpy(), dtype=torch.float32
    )
    
    # Save tensors to files
    with open(f"{base_path}train_batch_tensors.pkl", "wb") as f:
        pickle.dump(train_batch_tensors, f)
    
    with open(f"{base_path}train_tensor_all.pkl", "wb") as f:
        pickle.dump(train_tensor_all, f)

    with open(f"{base_path}valid_tensor_all.pkl", "wb") as f:
        pickle.dump(valid_tensor_all, f)

    return train_batch_tensors, train_tensor_all, valid_tensor_all

def prepare_padded_daily_data(batch_tensor, window_size=10):
    """
    Preprocess daily data by adding padding for each symbol
    Args:
        batch_tensor: Tensor containing one day's data
        window_size: Size of the sliding window
    Returns:
        Padded tensor with additional time steps for each symbol
    """
    device = batch_tensor.device

    # Group by symbol_id and sort by time_id
    symbols = batch_tensor[:, -3].unique()
    padded_data = []

    for symbol in symbols:
        # Get and sort data for current symbol
        symbol_data = batch_tensor[batch_tensor[:, -3] == symbol]
        symbol_data = symbol_data[torch.argsort(symbol_data[:, -2])]

        # Create padding data
        padding = torch.zeros((window_size-1, symbol_data.shape[1]), device=device)
        padding[:, -3] = symbol  # Set symbol_id in padding

        # Concatenate padding and actual data
        padded_symbol_data = torch.cat([padding, symbol_data], dim=0)
        padded_data.append(padded_symbol_data)

    return torch.cat(padded_data, dim=0)

def prepare_padded_valid_data(valid_tensor_all, window_size=10):
    """
    Preprocess validation data by adding padding
    Args:
        valid_tensor_all: Complete validation tensor
        window_size: Size of the sliding window
    Returns:
        Dictionary of padded validation tensors by date
    """
    # Group by date_id
    valid_dates = valid_tensor_all[:, -1].unique()
    valid_tensors_padded = {}
    
    for date in valid_dates:
        # Get and pad daily data
        daily_tensor = valid_tensor_all[valid_tensor_all[:, -1] == date]
        padded_tensor = prepare_padded_daily_data(daily_tensor, window_size)
        valid_tensors_padded[date] = padded_tensor
        
    return valid_tensors_padded

class Custom_Args:
    """Class to store default hyperparameters and configuration"""
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
        self.n_cont_features = 88
        self.n_cat_features = 2
        self.n_classes = None
        self.cat_cardinalities = [60, 1000]
        self.patience = 7
        self.max_epochs = 10
        self.N_fold = 5

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load_path', type=str, default=None, help='Path to load model checkpoint')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--small_batch_size', type=int, default=8192, help='Small batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--all_data', type=int, default=0, help='Use all data or not')
    parser.add_argument('--window_size', type=int, default=10, help='time window size')
    args = parser.parse_args()

    print(args)

    # Get training and validation tensors
    print("get 3 tensors ...")
    train_batch_tensors, train_tensor_all, valid_tensor_all = get_batch_tensors(saved=True, all_data=args.all_data)
    print("completed.")
    unique_dates = sorted(list(train_batch_tensors.keys()))

    # Prepare padded data
    print("prepare padded train")
    train_batch_tensors_padded = {}
    for date, tensor in train_batch_tensors.items():
        train_batch_tensors_padded[date] = prepare_padded_daily_data(tensor, args.window_size)

    print("prepare padded valid")
    valid_tensors_padded = prepare_padded_valid_data(valid_tensor_all, args.window_size)

    # Initialize or load model
    if args.load_path:
        print("load model:", args.load_path)
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

    # Setup optimizer and loss function
    optimizer = torch.optim.AdamW(
        make_parameter_groups(model),
        lr=args.lr,
        weight_decay=5e-3,
    )
    loss_fn = R2Loss()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Train model
    train_model(
        model,
        optimizer,
        loss_fn,
        batch_tensors=train_batch_tensors_padded,
        train_tensor_all=train_tensor_all,
        valid_tensor_all=valid_tensors_padded,
        unique_dates=unique_dates,
        num_epochs=args.num_epochs,
        small_batch_size=args.small_batch_size,
        window_size=args.window_size,
        save_dir=args.save_dir
    )