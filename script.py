from config import *
from core.utils import *

def train_model():
    # model = KiloNerf(N=32).to(device)
    model = torch.load(model_path)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    train(model, model_optimizer, scheduler, train_data_loader, nb_epochs=EPOCHS, device=device, hn=HN, hf=HF, nb_bins=NBINS)

def test_model():
    testing_dataset = torch.from_numpy(np.load(test_data_path, allow_pickle=True))
    model = torch.load(model_path)
    for idx in range(200):
        test(HN, HF, testing_dataset, model, device, img_index=0, nb_bins=NBINS, H=400, W=400)


if __name__ == '__main__':
    if device == 'cuda':
        train_model()
        test_model()