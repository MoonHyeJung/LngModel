import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt  # 플롯을 위해 matplotlib 추가

def train(model, trn_loader, device, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trn_loader):
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} - Training batch {batch_idx}/{len(trn_loader)}")
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0)).to(device)  # `to(device)` 추가
        outputs, _ = model(inputs, hidden)
        loss = criterion(outputs, targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} - Batch {batch_idx}/{len(trn_loader)}, Loss: {loss.item()}")

    trn_loss = total_loss / len(trn_loader)  # 평균 손실 값 계산
    return trn_loss

def validate(model, val_loader, device, criterion, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} - Validating batch {batch_idx}/{len(val_loader)}")
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0)).to(device)  # `to(device)` 추가
            outputs, _ = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} - Validation Batch {batch_idx}/{len(val_loader)}, Loss: {loss.item()}")

    val_loss = total_loss / len(val_loader)
    return val_loss

def plot_losses(train_losses, val_losses, model_type):  
    # 학습 및 검증 손실 값을 플롯하는 함수 추가
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_type} Training and Validation Loss')
    plt.legend()
    plt.show()

def main():
    print("Starting main function")

    # Hyperparameters
    batch_size = 64  # Increase batch size for better GPU utilization
    seq_length = 30
    hidden_size = 128
    num_layers = 2
    learning_rate = 0.005  # Adjust learning rate for faster convergence
    num_epochs = 10
    model_types = ['RNN', 'LSTM']  # RNN과 LSTM 모델을 각각 학습시키기 위해 리스트로 저장

    for model_type in model_types:
        print(f"Training {model_type} model")
        # Load dataset
        dataset = Shakespeare('shakespeare_train.txt')
        print(f"Dataset loaded with {len(dataset)} samples")

        # Split dataset into training and validation sets
        dataset_size = len(dataset)
        print(f"Dataset size: {dataset_size}")
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        split = int(np.floor(0.2 * dataset_size))
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)
        print("DataLoaders created")

        # Select device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Instantiate model
        vocab_size = len(dataset.chars)
        output_size = vocab_size
        if model_type == 'RNN':
            model = CharRNN(vocab_size, hidden_size, output_size, num_layers).to(device)
        else:
            model = CharLSTM(vocab_size, hidden_size, output_size, num_layers).to(device)
        print("Model instantiated")

        # Instantiate optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        print("Optimizer instantiated")

        # Instantiate cost function
        criterion = nn.CrossEntropyLoss()
        print("Criterion instantiated")

        train_losses = []  # 학습 손실 값을 저장할 리스트
        val_losses = []  # 검증 손실 값을 저장할 리스트

        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch+1}")
            train_loss = train(model, train_loader, device, criterion, optimizer, epoch+1)
            val_loss = validate(model, val_loader, device, criterion, epoch+1)

            train_losses.append(train_loss)  # 학습 손실 값 저장
            val_losses.append(val_loss)  # 검증 손실 값 저장

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            
            # Save the model with the best validation loss and vocab_size
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'vocab_size': vocab_size
                }
                torch.save(checkpoint, f'best_{model_type.lower()}_model.pth')
                print(f"Model saved at epoch {epoch+1}")

        # Plot the training and validation losses
        plot_losses(train_losses, val_losses, model_type)  # 손실 값 플롯

if __name__ == '__main__':
    main()
