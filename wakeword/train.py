from pathlib import Path
from sklearn.metrics import f1_score
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from wakeword.dataset import WakeWordDataSet
from wakeword.model import WakeWordDetector
from wakeword.utils import collate_fn

train_data = WakeWordDataSet(Path('./wakeword/train.csv'), 8000)
test_data = WakeWordDataSet(Path('./wakeword/test.csv'), 8000)
model = WakeWordDetector(40, 64, num_layers=4)

train_loader = DataLoader(dataset=train_data, batch_size=32, collate_fn=collate_fn, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=4, collate_fn=collate_fn)

epochs = 20

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)

opt = optim.AdamW(model.parameters(), lr=0.001)
best_f1 = -1
criterion = torch.nn.BCELoss()
for epoch in range(epochs):
    running_loss = 0.0
    running_f1 = 0.0
    model.train(True)
    for i, data in enumerate(train_loader):
        mfccs = data[0].to(device)
        labels = data[1].to(device)

        opt.zero_grad()
        outputs = model(mfccs)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)
        rounded = torch.round(outputs)
        loss.backward()
        opt.step()
        running_f1 += f1_score(labels.cpu().numpy(), rounded.cpu().detach().numpy())
        running_loss += loss.item()
        if i % 10 == 9:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            print('[%d, %5d] f1: %.3f' %
                  (epoch + 1, i + 1, running_f1 / 10))
            running_loss = 0.0
            running_f1 = 0.0

    test_f1 = 0.0
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            mfccs = test_data[0].to(device)
            labels = test_data[1].to(device)

            outputs = model(mfccs)
            outputs = torch.sigmoid(outputs)
            rounded = torch.round(outputs)
            current_f1 = f1_score(labels.cpu().numpy(), rounded.cpu().detach().numpy())
            test_f1 += current_f1
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        print(f'test loss: {test_loss / len(test_loader)}')
        print(f'test f1: {test_f1 / len(test_loader)}')
        if test_f1 / len(test_loader) > best_f1:
            torch.save(model.state_dict(), './models/wakeword_checkpoint.pt')
            best_f1 = test_f1 / len(test_loader)
            # break
