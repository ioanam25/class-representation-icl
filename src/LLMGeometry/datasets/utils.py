from torch.utils.data import DataLoader

def get_dataloader(dataset, BATCH_SIZE=32, shuffle=True):
    return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=lambda x: x, shuffle=shuffle)