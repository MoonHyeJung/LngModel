import torch
from torch.utils.data import Dataset, DataLoader

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
           You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):
        # Load input file
        with open(input_file, 'r') as f:
            self.text = f.read()

        # Create character dictionary
        self.chars = sorted(set(self.text))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}

        # Make list of character indices
        self.data = [self.char_to_idx[char] for char in self.text]

        # Split the data into chunks of sequence length 30
        self.seq_length = 30
        self.data_size = len(self.data) - self.seq_length

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx+self.seq_length]
        target_seq = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(input_seq), torch.tensor(target_seq)

if __name__ == '__main__':
    # Test codes to verify the implementation
    dataset = Shakespeare('shakespeare_train.txt')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for input_seq, target_seq in dataloader:
        print("Input sequence:", input_seq)
        print("Target sequence:", target_seq)
        break
