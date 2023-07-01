import argparse
from models import *
from dataset import BaselineDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=200)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", type=str)
args = parser.parse_args()

if args.gpu >= 0 and type(args.gpu) == int:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

test_data = BaselineDataset(train=False)
TestLoader = DataLoader(test_data, batch_size=args.batchsize, shuffle=False)

model = torch.load(args.checkpoint)
model.to(device)

def CrossEntropy(target, prediction):
    prob = prediction.softmax(dim=-1)
    log_prob = torch.log(prob + 1e-6)
    entropy =  target * log_prob
    loss = - entropy.sum(-1).mean()
    return loss 

model.eval()
total_test_loss = 0
total_accuracy = 0 
with torch.no_grad():
    for data in TestLoader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = CrossEntropy(targets, outputs)

        total_test_loss = total_test_loss + loss.item()
        accuracy = (outputs.argmax(1) == targets.argmax(1)).sum()
        total_accuracy = total_accuracy + accuracy

print("Total loss:{}".format(total_test_loss))
print("Accuracy:{}".format(total_accuracy/len(test_data)))