import torch
import torch.nn as nn
from torch import optim
from src import WasteImageNN
from src import WasteImageNNLight
from src import plot_training_loss
from src import train_loader, test_loader, val_loader
from src import evaluate_test
from pathlib import Path
import time 

def train(model, train_loader, val_loader, num_ephohs, lr):
    """
    Train model and track loss

    Args:

        model : Pytorch model
        train_loader : Training Data loader
        num_epohs : # of Epochs
        lr : learning rate

    returns:

        training losses: list of traing losses per epoch
    """
    # check the gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using the device : {device}\n")

    # move the model to device
    model = model.to(device)

    # loss and optimizer
    cirterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss = []
    validation_loss = []

    print(f"Training for {num_ephohs} epoches...\n")

    for epoch in range(num_ephohs):
        # ============================= Training =============================
        model.train()
        epoch_loss = 0

        for batch_x, batch_y in train_loader:
            # move batch to the device 
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # zero gradients 
            optimizer.zero_grad()

            # forward pass 
            outputs = model(batch_x)
            loss = cirterion(outputs, batch_y)

            # backpropagation
            loss.backward()

            # update weights 
            optimizer.step()

            epoch_loss += loss.item()

        # avarage training loss 
        avg_loss = epoch_loss/len(train_loader)
        train_loss.append(avg_loss)

        # ============================= Validation =============================
        model.eval()
        valid_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = cirterion(outputs, batch_y)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss/len(val_loader)
        validation_loss.append(avg_valid_loss)


        print(f"Epoch [{epoch+1}/{num_ephohs}] | Train Loss: {avg_loss:.4f} |  Validation Loss : {avg_valid_loss:.4f}")

    print("\nTrain is Completed !")
    return train_loss, validation_loss


if __name__ == "__main__":
    # CNN - Model A
    model = WasteImageNN()
    start_time_model_A = time.perf_counter()
    train_loss, validation_loss = train(model=model, train_loader=train_loader, val_loader=val_loader, num_ephohs=20, lr=1e-3)
    end_time_model_A = time.perf_counter()
    print(f"Training time : {end_time_model_A-start_time_model_A}")
    plot_training_loss(train_losses=train_loss, validation_losses=validation_loss)
    # save model
    torch.save(model.state_dict(), Path("../models/model_A.pth"))
    print(f"Model Saved Successfully")
    # evaluvate test data
    test_acc = evaluate_test(model=model, test_loader=test_loader)

    # Depthwise Separable CNN
    model_light = WasteImageNNLight()
    start_time_model_B = time.perf_counter()
    train_loss_, validation_loss_ = train(model=model, train_loader=train_loader, val_loader=val_loader, num_ephohs=20, lr=1e-3)
    end_time_model_B = time.perf_counter()
    plot_training_loss(train_losses=train_loss_, validation_losses=validation_loss_)
    # save model
    torch.save(model.state_dict(), Path("../models/model_B_light.pth"))
    print(f"Model Saved Successfully")
    # evaluvate test data
    test_acc = evaluate_test(model=model_light, test_loader=test_loader)
