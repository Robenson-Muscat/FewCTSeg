##### TO DENORMALIZE ########


# Correctly classified
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

# Flatten axes for easy iteration
axes = axes.flatten()

selected_indices = correct_images[:10]

# Iterate through the selected indices
for i, idx in enumerate(selected_indices):
    image, label = test_loader.dataset[idx]  # Extract image and label

    # If image is RGB, swap dimensions from (C, H, W) to (H, W, C)
    mean=torch.tensor([.5])
    std=torch.tensor([.5])
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    img = np.clip(unnormalize(image).numpy(),0,1)
    img = img.transpose(1, 2, 0)  # Convert from PyTorch format to Matplotlib format

    axes[i].imshow(img)  # Adjust cmap if necessary
    axes[i].set_title(f"True Label: {label[0]}\nPredicted label:{pred[idx]}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()


###### OTHER CONFIGURATION FOR SEGMENTATION MASK####
from torchvision.transforms import v2
# 1.bis Define transforms
train_transform = v2.Compose([
     v2.CenterCrop(size=(128,128)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        v2.ToTensor()
    ],
)

val_transform = v2.Compose([
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    v2.ToTensor(),
],)


# Changing transform in CTScanDataset
        if self.transform:
            # Pass both image and mask
            #augmented = self.transform(image=img, mask=mask)
            #img_tensor = augmented['image']
            #mask_tensor = augmented['mask']
            img_tensor = self.transform(img)
            mask_tensor = self.transform(mask)
