import torch
from model import *
from dataset import WIDERFaceDataset
from tqdm import tqdm
from utils import *

data_folder = './PyramidBox' 
batch_size = 16
weight_decay = 0.0005
momentum = 0.9
lr = 1e-3
num_epochs = 10
print_freq = 200


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PyramidBox()    
    model = model.to(device)
    model.train()

    train_dataset = WIDERFaceDataset(data_folder, split = 'Train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               collate_fn=train_dataset.collate_fn)

    criterionFace = PyramidBoxLoss(model.face_prior_cxcy).to(device)
    criterionHead = PyramidBoxLoss(model.head_prior_cxcy, supervised=False).to(device)

    optimizer = torch.optim.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1} / {num_epochs}")

        for i, (images, boxes, face_targets, head_targets) in loop:
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]

            face_locs, head_locs, face_scores, head_scores = model(images)

            face_loc_loss, face_conf_loss = criterionFace(face_locs, face_scores)
            head_loc_loss, head_conf_loss = criterionHead(head_locs, head_scores)

            loss = face_loc_loss + face_conf_loss + head_loc_loss + head_conf_loss
            
            optimizer.zero_grad()
            loss.backward()

            loop.set_postfix(loss=loss.item())

            # if i % print_freq == 0:
            #     print('Epoch: [{0}][{1}/{2}]\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader), loss=losses))

            loop.update(1)
        
        save_checkpoint(epoch, model, optimizer)

if __name__ == '__main__':
    train()