from tkinter import Variable
import torch
from Param import *
def predict(model, data, file, map_reverse, testNum):
    model.eval()
    print("\n-----------Start testing!-----------")
    with open(file, 'a') as file:
        print("\n-----------Start testing!-----------", file=file)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                shuffle=True, num_workers=0)
        total = 0.0
        correct = 0.0
        for loader in data_loader:
            for images, labels in loader:
                # images = Variable(images).cuda()
                preds = model.classify(images)
                preds = [map_reverse[pred] for pred in preds.cpu().numpy()]
                total += labels.size(0)
                correct += (preds == labels.numpy()).sum()

        # Accuracy
        print (f'Test data number = {testNum}')
        print (f'Test data number = {testNum}', file=file)
        print ('Accuracy : %.2f ,' % (100.0 * correct / total))
        print ('Accuracy : %.2f ,' % (100.0 * correct / total), file=file)
