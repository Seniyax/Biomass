def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for left_imgs, right_imgs, train_targets, all_targets in dataloader:
        left_imgs = left_imgs.to(device)
        right_imgs = right_imgs.to(device)
        targets = train_targets.to(device)

        optimizer.zero_grad()

        prediction = model(left_imgs, right_imgs)
        loss = criterion(prediction, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() 

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss