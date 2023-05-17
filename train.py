import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5], std=[0.5])]
    )
    train_set = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,
                                               shuffle=True, num_workers=0)
    val_set = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_images, val_labels = next(val_data_iter)
    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    for epoch in range(5):

        running_loss = 0.0

        for step, (images, labels) in enumerate(train_loader, start=0):
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 500 == 499:
                with torch.no_grad():
                    outputs = net(val_images)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_labels).sum().item() / val_labels.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0
    print('Finished Training')

    save_path = './Lenet05.pth'
    torch.save(net.state_dict(), save_path)


if __name__=='__main__':
    main()
