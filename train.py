
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from torch.utils.data import DataLoader, TensorDataset
from model.dnn import DNN
import mlflow
import mlflow.pytorch
import datetime


DATASET_PATH = "data/dataset.csv"
EPOCHS = 1

# Get cpu or gpu device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def get_data():
    # Load the data from csv file into a pandas DataFrame.
    df = pd.read_csv(DATASET_PATH)

    # Normalize dataset to have all the values in the same range.
    mins = df.min()
    maxs = df.max()
    df = (df-mins)/(maxs-mins)

    # Separate features and target.
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Split data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prepare data for training the model.
    batch_size = 100

    tensor_X_train = torch.Tensor(X_train).to(device)
    tensor_y_train = torch.Tensor(y_train).to(device)

    tensor_X_test = torch.Tensor(X_test).to(device)
    tensor_y_test = torch.Tensor(y_test).to(device)

    train_dataset = TensorDataset(tensor_X_train,tensor_y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(tensor_X_test,tensor_y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)

    model.train()
    preds = torch.Tensor().to(device)
    targets = torch.Tensor().to(device)
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        # Compute prediction error.
        pred = model(X)
        pred = pred.squeeze(1)
        loss = loss_fn(pred, y)

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = torch.cat((preds, pred))
        targets = torch.cat((targets, y))

    # Check if preds contains any NaN values.
    if torch.isnan(preds).any():
        print("Warning: preds contains null (NaN) values!")

    targets = targets.detach().cpu()
    preds = torch.nan_to_num(preds.detach().cpu())

    # Calculate metrics.
    mse = mean_squared_error(targets, preds)
    targets_np = targets.cpu().numpy()
    preds_np = preds.cpu().numpy()
    r2 = r2_score(targets_np, preds_np)
    mape = mean_absolute_percentage_error(targets, preds)

    return r2, mape, mse

def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    preds = torch.Tensor().to(device)
    targets = torch.Tensor().to(device)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Compute prediction error.
            pred = model(X)

            preds = torch.cat((preds, pred))
            targets = torch.cat((targets, y))

    preds = preds.squeeze()
    # Check if preds contains any NaN values.
    if torch.isnan(preds).any():
        print("Warning: preds contains null (NaN) values!")

    targets = targets.detach().cpu()
    preds = torch.nan_to_num(preds.detach().cpu())

    # Calculate metrics.
    mse = mean_squared_error(targets, preds)
    targets_np = targets.cpu().numpy()
    preds_np = preds.cpu().numpy()
    r2 = r2_score(targets_np, preds_np)
    mape = mean_absolute_percentage_error(targets, preds)

    return r2, mape, mse


def run_train_test_process(
        model,
        loss_fn,
        optimizer,
        train_dataloader,
        test_dataloader,
        hidden_layers_num,
        learning_rate,
        optimizer_name,
        epochs=20,
):
    best_test_mse = None

    print(
        "-"*70 + "\n"
        f"Starting training model: "
        f"[{hidden_layers_num}] hidden layers | "
        f"[{learning_rate}] learning-rate | "
        f"[{optimizer_name}] optimizer..."
    )
    for epoch in range(1, epochs + 1):
        train_r2, train_mape, train_mse = train(train_dataloader, model, loss_fn, optimizer)
        test_r2, test_mape, test_mse = test(test_dataloader, model)
        
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("train_mape", train_mape)
        mlflow.log_metric("train_mse", train_mse)
        
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_metric("test_mape", test_mape)
        mlflow.log_metric("test_mse", test_mse)

        best_test_mse = min(test_mse, best_test_mse) if best_test_mse else test_mse

    return best_test_mse


def run_experiment(hidden_layers_num, learning_rate, train_dataloader, test_dataloader):
    models = [
        DNN(hidden_layers_num).to(device)
        for _ in range(3)
    ]
    optimizers = [
        ("SGD", torch.optim.SGD(models[0].parameters(), lr=learning_rate)),
        ("RMSProp", torch.optim.RMSprop(models[1].parameters(), lr=learning_rate)),
        ("Adam", torch.optim.Adam(models[2].parameters(), lr=learning_rate))
    ]

    best_test_mse_results = []
    for model, (optimizer_name, optimizer) in zip(models, optimizers):
        with mlflow.start_run():
            loss_fn = nn.MSELoss()
            best_test_mse = run_train_test_process(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                hidden_layers_num=hidden_layers_num,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                epochs=EPOCHS,
            )
            mlflow.pytorch.log_model(model, f"model-{hidden_layers_num}-{learning_rate}-{optimizer_name}")
            mlflow.log_param("hidden_layers", hidden_layers_num)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("optimizer", optimizer_name)

            best_test_mse_results.append((
                hidden_layers_num,
                learning_rate,
                optimizer_name,
                best_test_mse,
                mlflow.active_run().info.run_id
            ))
            print(f"Best Test MSE: [{best_test_mse}]")

    return best_test_mse_results


def main():
    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mlflow.set_experiment(experiment_name)
    print(f"Experiment: {experiment_name}")

    train_dataloader, test_dataloader = get_data()
    print(f"Train dataset size: {len(train_dataloader.dataset)}")
    hidden_layers_nums = [5]
    learning_rates = [0.001]

    best_test_mse_results = []
    for hidden_layers_num in hidden_layers_nums:
        for learning_rate in learning_rates:
            best_test_mse_results.extend(
                run_experiment(
                    hidden_layers_num=hidden_layers_num,
                    learning_rate=learning_rate,
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                )
            )

    best_test_mse_results = sorted(best_test_mse_results, key=lambda x: x[-1])
    print("\n\nBest Test MSE Results:")
    for hidden_layers_num, learning_rate, optimizer_name, best_test_mse in best_test_mse_results:
        print(f"Hidden Layers: [{hidden_layers_num}] | Learning Rate: [{learning_rate}] | "
              f"Optimizer: [{optimizer_name}] | Best Test MSE: [{best_test_mse}]")

    # Deploy the best model.
    best_model_info = best_test_mse_results[0]
    best_model_run_id = best_model_info[-1]
    best_model_path = f"model-{best_model_info[0]}-{best_model_info[1]}-{best_model_info[2]}"

    # URI of the best model
    model_uri = f"runs:/{best_model_run_id}/{best_model_path}"
    print(f"Best model URI: {model_uri}")

    # To serve the best model
    mlflow.pyfunc.serve(model_uri=model_uri, host="127.0.0.1", port=1234)
    

if __name__ == "__main__":
    main()