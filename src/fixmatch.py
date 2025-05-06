# Define a function that weakly augments the images
def weak_augment(images,masks):
    augment= A.Compose([
    A.Transpose(p=1), # Ou verticalFlip
    A.CLAHE(clip_limit=(4,4),p=1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.ToTensorV2(),
    ],)
    augmented = augment(image=img, mask=mask)
    images = augmented['image']
    masks = augmented['mask']
    return images, masks
  

# Define a function that very strongly augments the images
def strong_augment(images,masks):
    augment= A.Compose([
        A.Transpose(p=1),
        A.ElasticTransform(alpha=400, sigma=10, alpha_affine=None, p=1),
        A.GridDistortion(distort_limit=0.8, num_steps = 5, p=1),#-0.8,0.8
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),])
    augmented = augment(image=img, mask=mask)
    images = augmented['image']
    masks = augmented['mask']
    return images, masks
    


# Define a function that creates pseudo labels above a threshold of confidence and assign -1 to the others
def create_pseudo_labels(model, images, threshold):
####TO COMPLETE###
    

# Data

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

full_dataset = CTScanDataset(
    image_dir=train_path,
    mask_csv=labels_path,
    transform=None
)

# Split into train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = random_split(
    range(len(full_dataset)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(26)
)

# Create train and val datasets
train_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2(),
    ],
)
train_lab = LabCTScanDataset(image_dir=train_path, mask_csv=labels_path,transform=train_transform) #800 images
train_unlab = UnLabCTScanDataset(image_dir=image_dir, mask_csv=mask_csv,transform = None) # 1200 images

val_ds = LabCTScanDataset(
    image_dir=train_path,
    mask_csv=labels_path,
    transform=train_transform
)


train_ds = torch.utils.data.Subset(train_ds, train_indices.indices) #640 images
val_ds = torch.utils.data.Subset(val_ds, val_indices.indices) #160 images



# FixMatch Hyperparameter 
lambda_u = 1 # loss weight
tau = 0.95 # weakly augmented threshold

# Training parameters
epochs = 60
lr = 1e-3
bs_lab = 8 # lab batch size
bs_unlab = 15 # unlab batch size
bs_total = bs_lab + bs_unlab # total batch size
steps_per_epoch = math.floor(len(train_lab)/bs_lab) #640/8=80
prev_acc = 0

# Logs
train_acc = []
test_acc = []
losses = []
losses_lab = []
losses_unlab = []
pseudo_lab = []

# Create model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=55,
    activation=None
).to(DEVICE)

# Training settings
loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# metrics
train_acc_metric = ...
test_acc_metric = ...

# indexes
indices_lab = np.arange(len(train_lab)) # lab data
indices_unlab = np.arange(len(train_unlab)) # unlab data

val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)
# Training loop
for epoch in range(epochs):

  # Shuffle the lab data
  train_lab_loader = DataLoader(train_lab, batch_size=bs_lab, shuffle=True, num_workers=4)

  # Shuffle the unlab data
  train_unlab_loader = DataLoader(train_unlab, batch_size=bs_unlab, shuffle=True, num_workers=4)

  # Initialize cumulative losses
  cumul_loss_value = 0
  cumul_loss_value_lab = 0
  cumul_loss_value_unlab = 0

  # Intialize the number of pseudo labels
  num_pseudo_labels = 0

  # Training 
    
  # Get the lab batch 
    
  # Get the unlab batch
    
  # Get predictions for the lab batch
  ...

  # Get predictions for the unlab batch (weakly augmented)
  ...

  # Get predictions for the unlab batch (strongly augmented)
  ...    

  # Compute the loss for valid pseudo labels
  ...

  # Compute the loss for the unlab batch if there are valid pseudo labels
  ...

  #Compute the total loss
  loss_value = loss_value_lab + lambda_u * loss_value_unlab
  ...

  # Compute the gradients
    ...
    

  # Update the weights
    ...
    

  # Update the cumulative loss
    ...

  # Update the training accuracy
    ...

  
  # Calculate the average pseudo labels
  ...

  # Calculate the  training dice
  ...

  # Reset the training metrics
  ...

  # Calculate the validation dice
  
  # Evaluate the model on the validation data 
  ...

  # Calculate the validation dice
  ...

  # if the validation dice is better than the previous one or if epoch%10==0, save the model
  ...

  # Reset the test metrics
  ...

  # Save the results
  ...

  # Print the results (loss, loss_lab, loss_unlab, dice_train, dice_val and pseudo labels)
  ...