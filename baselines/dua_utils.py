import torch


def tensor_rot_90(x):
    return x.flip(3).transpose(2, 3)


def tensor_rot_180(x):
    return x.flip(3).flip(2)


def tensor_rot_270(x):
    return x.transpose(2, 3).flip(3)


def rotate_batch_with_labels(batch, labels):
    images = []
    for img, label in zip(batch, labels):
        if label == 1:
            img = tensor_rot_90(img)
        elif label == 2:
            for image in batch:
                img = tensor_rot_180(image)
        elif label == 3:
            for image in batch:
                img = tensor_rot_270(image)
        images.append(img)
    return torch.stack(images)


def rotate_batch(batch): # input [b, channel, frames, h, w] if arch != tanet else [b, frames, channels, h, w]

    labels = torch.randint(4, (len(batch),), dtype=torch.long)
    list_batch = [batch[i, :, :, :, :] for i in range(len(batch))] # make a list of tensors
    return rotate_batch_with_labels(list_batch, labels)
