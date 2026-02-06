import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report


data = np.load("preprocessed-galaxy-classification-36x36.npz")
X = data['X']
Y_ratio = data['Y_ratio']
Y_time = data['Y_time']
Y_is_merger = data['Y_is_merger']
Y_stellar_mass = data['Y_stellar_mass']

print(f"{np.count_nonzero(Y_is_merger)} mergers | {len(Y_is_merger) - np.count_nonzero(Y_is_merger)} non mergers")

'''
fig = plt.figure(figsize=(10, 7))

# Add the first image to the figure (top-left position)
plt.subplot(1, 3, 1)  # 2 rows, 2 columns, first position
plt.imshow((X[42].transpose(1, 2, 0)))  
plt.axis('off')  # Hide the axis labels
plt.title("Image 1") 

# Add the second image to the figure (top-right position)
plt.subplot(1, 3, 2)  # 2 rows, 2 columns, second position
plt.imshow((X[42].transpose(1, 2, 0)))  
plt.axis('off')  # Hide the axis labels
plt.title("Image 2") 

# Add the third image to the figure (bottom-left position)
plt.subplot(1, 3, 3)  # 2 rows, 2 columns, third position
plt.imshow((X[42].transpose(1, 2, 0))) 
plt.axis('off')  # Hide the axis labels
plt.title("Image 3")  

plt.show()
'''


random_state = 42

# SOURCE: Mathilda Avirett-Mackenzie, modified by Alex Brown
def get_random_d4_augs(img, n_aug=1, replace=True, seed=None):
    img = np.transpose(img, (1, 2, 0))  # (C, M, N) to (M, N, C)
    rng = np.random.default_rng(seed=seed)

    # whether to reflect and number of rotations
    aug_list = [(False, 0), (True, 0), (False, 1), (True, 1),
                (False, 2), (True, 2), (False, 3), (True, 3)]

    # forces replacement if more than 8 augmentations are requested
    if n_aug > 8:
        replace = True

    augs_used = rng.choice(aug_list, size=n_aug, replace=replace)

    output = []

    for (reflect, n_rot) in augs_used:
        # initialise image to add to output list
        add_img = img

        if reflect:
            add_img = np.flip(add_img, axis=0)

        for i in range(n_rot):
            add_img = np.rot90(add_img, axes=(0, 1))

        add_img = np.transpose(add_img, (2, 0, 1))  # (M, N, C) to (C, M, N)
        output.append(add_img)

    return output


# SOURCE: Mathilda Avirett-Mackenzie
def get_n_aug(log_mass):
    if log_mass < 10:
        return 1
    if log_mass < 11:
        return 2
    if log_mass < 11.5:
        return 3
    return 8


# SOURCE: Alex Brown
def boost(data, target, y_stellar):
    X_boosted, Y_merger_boosted, Y_stellar_boosted = [], [], []
    for i, img in enumerate(data):
        log_mass = np.log10(y_stellar[i]) + 10
        n_aug = get_n_aug(log_mass)
        aug_imgs = get_random_d4_augs(img, n_aug)
        for aug_img in aug_imgs:
            X_boosted.append(aug_img)
            Y_merger_boosted.append(target[i])
            Y_stellar_boosted.append(y_stellar[i])

    return np.array(X_boosted), np.array(Y_merger_boosted), np.array(Y_stellar_boosted)


train_transform = T.Compose([
    T.ToTensor(), 
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15),
    T.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05), 
        scale=None,
        shear=None,
        fill=0
    )
])

# Validation/Test sets should not be augmented
test_transform = T.Compose([
    T.ToTensor()
])

class GalaxyDataset(Dataset):
    def __init__(self, X_data, y_data, transform=None):
        self.X_data = X_data.transpose(0, 2, 3, 1)
        self.y_data = y_data
        self.transform = transform

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = self.X_data[idx]
        label = self.y_data[idx]

        if self.transform:
            img = self.transform(img)
            
        return img.float(), torch.tensor(label).long()


Y_stacked = np.stack([Y_is_merger, Y_stellar_mass], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, Y_stacked, test_size=0.2, stratify=Y_is_merger, random_state=random_state)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test[:, 0], random_state=random_state)

y_stellar_train_original = y_train[:, 1].copy()
y_stellar_val_original = y_val[:, 1].copy()

y_stellar_train = y_train[:, 1]
y_train = y_train[:, 0]

y_stellar_val = y_val[:, 1]
y_val = y_val[:, 0]

y_test = y_test[:, 0]

print("Data split")

X_train, y_train, y_train_stellar = boost(X_train, y_train, y_stellar_train)
X_val, y_val, y_val_stellar = boost(X_val, y_val, y_stellar_val)

print("Data boosted")


train_dataset = GalaxyDataset(X_train, y_train, transform=train_transform)
val_dataset = GalaxyDataset(X_val, y_val, transform=test_transform) 
test_dataset = GalaxyDataset(X_test, y_test, transform=test_transform) 


BATCH_SIZE = 64 
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Datasets and Loader created")
 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 6)
        self.batch1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.batch2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 4)
        self.batch3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, 3)
        self.batch4 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.1)


        self.fc1 = nn.Linear(12800, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):

        x = self.dropout(self.batch1(F.relu(self.conv1(x))))
        x = self.dropout(self.batch2(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(self.batch3(F.relu(self.conv3(x)))))
        x = self.dropout(self.batch4(F.relu(self.conv4(x))))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print()

N_EPOCHS = 50
print(f"Starting training for {N_EPOCHS} epochs...")

for epoch in range(N_EPOCHS):
    running_loss = 0.0
    net.train() # Set model to training mode
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # --- Validation Loop ---
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    net.eval() # Set model to evaluation mode
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = net(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f'Epoch [{epoch + 1}/{N_EPOCHS}] | '
          f'Train Loss: {running_loss / len(trainloader):.3f} | '
          f'Val Loss: {val_loss / len(valloader):.3f} | '
          f'Val Acc: {val_accuracy:.2f} %')

print('Finished Training')

# --- 8. Final Evaluation on Test Set ---
print("Evaluating on Test Set...")
net.eval()
all_labels = []
all_predicted = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())


print("\n--- Test Set Classification Report ---")
print(classification_report(all_labels, all_predicted, target_names=['Non-Merger', 'Merger']))
print("--------------------------------------")