import os, re, glob, json
import numpy as np, pandas as pd, cv2
import torch, random, copy, math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch.nn.functional as F

from utils import alphanumeric_sort, set_seed, compute_per_class_dice
from src.dataset import LabeledCTScanDataset, UnlabeledPathsDataset, TestCTScanDataset
from src.unimatch import inf_loop, pseudo_targets_from_logits, feature_perturbation, weak_perturbation, strong_perturbation
from src.post_processing import relabel_fp_by_convex_hull, inference_with_postprocessing




# ------------------ Configuration ------------------
SEED = 26
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = "./data/"
IMG_DIR = os.path.join(PATH, 'train-images/')
MASK_CSV = os.path.join(PATH, 'y_train.csv')
BATCH = 8
NUM_CLASSES = 55
NUM_EPOCHS_PHASE1 = 35
NUM_EPOCHS_PHASE2 = 30
LAMBDA = 0.5  # weight feature-perturb loss
MU = 0.5      # weight image-level loss
IGNORE_INDEX = 255  # to filter noisy labels
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

# For UniMatch
TAU = 0.95      # confidence threshold for pixel-wise masking
FP_DROP_P = 0.5  # Dropout2d probability on last feature map

# KFold settings
K_FOLDS = 5
SHUFFLE_BEFORE_K = True

#Filenames
CSV_FILENAME = 'submission_fewctseg.csv'

# ------------------ Utils ------------------
set_seed(SEED)


# ------------------ Transforms ------------------
base_transform = A.Compose([
    A.Normalize(mean=tuple(MEAN.tolist()), std=tuple(STD.tolist())),
    ToTensorV2()])

weak_transform = A.Compose([
    A.RandomResizedCrop((256,256), scale=(0.2,0.8), p=1.0),
    A.Normalize(mean=tuple(MEAN.tolist()), std=tuple(STD.tolist())),
    ToTensorV2()])


strong_transform = A.Compose([
    A.CoarseDropout(num_holes_range=(3,8),hole_height_range = (0.1,0.3), hole_width_range=(0.1,0.3),p=1.0),
    A.Normalize(mean=tuple(MEAN.tolist()), std=tuple(STD.tolist())),
    ToTensorV2()])

# ------------------ Datasets ------------------

full_lab = LabeledCTScanDataset(IMG_DIR, MASK_CSV, transform=None)
N_lab = len(full_lab)
print("Total labeled (non-empty masks) samples:", N_lab)

# indices for KFold (relative to filtered full_lab)
base_indices = np.arange(N_lab)
if SHUFFLE_BEFORE_K:
    rng = np.random.RandomState(SEED)
    rng.shuffle(base_indices)



# unlabeled dataset
paths_all = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')), key=alphanumeric_sort)
masks_all = pd.read_csv(MASK_CSV, index_col=0).T.values.reshape(-1,256,256).astype(np.uint8)
valid_unlab = [m.sum()==0 for m in masks_all]
unlab_paths = [p for p,v in zip(paths_all, valid_unlab) if v]
unlabeled_ds = UnlabeledPathsDataset(unlab_paths)
print("Total unlabeled images:", len(unlabeled_ds))


#TEST_HEAD_INDEX = np.load(os.path.join(PATH, "test_head_index.npy"))




# ------------------ Model factory ------------------
def create_model():
    return smp.Segformer(
        encoder_name='timm-efficientnet-b7', encoder_weights='imagenet',
        in_channels=3, classes=NUM_CLASSES
    ).to(DEVICE)

def load_model_from_file(model_path):
    model = create_model()  
    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict):
        
        if 'state_dict' in ckpt:
            state = ckpt['state_dict']
        else:
            state = ckpt
        try:
            model.load_state_dict(state)
        except RuntimeError:
            from collections import OrderedDict
            new_state = OrderedDict()
            for k, v in state.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state[name] = v
            model.load_state_dict(new_state)
    else:
    
        try:
            model = ckpt.to(DEVICE)
            model.eval()
            return model
        except Exception:
            raise ValueError("Unsupported checkpoint format.")
    model.to(DEVICE)
    model.eval()
    return model



# ------------------ Phase1 trainer ------------------
def train_phase1(model, train_ds, val_ds, epochs=NUM_EPOCHS_PHASE1, batch_size=BATCH, seed=SEED, model_out_dir=PATH, fold=0):
    gen = torch.Generator(); gen.manual_seed(seed + fold)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, generator=gen, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    opt = Adam(model.parameters(), lr=1e-3)
    sup_loss_local = smp.losses.DiceLoss(mode='multiclass', from_logits=True)

    best_val = 1e9
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for imgs, masks,_ in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            logits = model(imgs)
            loss = sup_loss_local(logits, masks.long())
            loss.backward(); opt.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks,_ in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_loss += sup_loss_local(model(imgs), masks.long()).item() * imgs.size(0)
        val_loss /= len(val_ds)
        history.append((train_loss, val_loss))
        print(f"[Fold {fold}] Phase1 Epoch {epoch}/{epochs} — train:{train_loss:.4f} val:{val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(model_out_dir, f'best_val_fold{fold}_phase1.pth'))
    return best_val, history

# ------------------ Phase2 trainer (unlabeled-driven) ------------------
def train_phase2(model, train_ds, unlabeled_ds, val_ds,
                                  num_epochs=NUM_EPOCHS_PHASE2, batch_size=BATCH,
                                  tau=TAU, fp_drop_p=FP_DROP_P,
                                  lambda_=LAMBDA, mu=MU,
                                  ignore_index=IGNORE_INDEX,
                                  model_out_dir=PATH, fold=0):
    #gen = torch.Generator(); gen.manual_seed(seed + fold)

    bs_lab = max(1, batch_size // 2)
    bs_unlab = max(1, batch_size - bs_lab)
    labeled_loader = DataLoader(train_ds, batch_size=bs_lab, shuffle=True, num_workers=0, drop_last=True) #To keep batch_size even
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=bs_unlab, shuffle=True, num_workers=0, drop_last=True) 
    val_loader_local = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    unlab_iter = inf_loop(unlabeled_loader)

    sup_loss_local = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    ce_local = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    drop2d_local = nn.Dropout2d(p=fp_drop_p)
    opt = Adam(model.parameters(), lr=1e-4)
    scheduler_local = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)

    best_val = 1e9
    best_dice = -1
    fractions_per_epoch = []
    history = []

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        fractions_this_epoch = []
        steps = 0

        loop = tqdm(labeled_loader, desc=f"[Fold {fold}] Phase2 E{epoch}", leave=False)
        for imgs_lab, masks_lab,_ in loop:
            steps += 1

            imgs_lab, masks_lab = imgs_lab.to(DEVICE), masks_lab.to(DEVICE).long()
            imgs_unlab, _,_ = next(unlab_iter)

            # supervised labeled forward
            logits_lab = model(imgs_lab)
            #supervised loss
            loss_sup = sup_loss_local(logits_lab, masks_lab)




            # Apply weak augmentation on raw unlabeled images, returns normalized tensor
            batch_w = weak_perturbation(imgs_unlab, weak_transform, DEVICE)
            #Apply then two differents strongs augmentations
            batch_s1, batch_s2 = strong_perturbation(batch_w, strong_transform, MEAN,STD, DEVICE)


            # feature perturbation on last feature map
            feat_w = model.encoder(batch_w)
            logits_w = model.segmentation_head(model.decoder(feat_w))
            logits_fp = feature_perturbation(model, feat_w,drop2d_local)

            # two strong predictions (image-level strong perturbations)
            logits_s1 = model(batch_s1)
            logits_s2 = model(batch_s2)

            # build pseudo targets (pixel-wise mask using TAU - confidence threshold)
            target_u, mask_conf = pseudo_targets_from_logits(logits_w, tau=TAU, ignore_index=IGNORE_INDEX)
            fraction_kept = float(mask_conf.float().mean().item())
            fractions_this_epoch.append(fraction_kept)


            # unsupervised losses
            p_s_cat = torch.cat([logits_s1, logits_s2], dim=0)
            target_s_repeat = torch.cat([target_u, target_u], dim=0)
            loss_s = ce_local(p_s_cat, target_s_repeat)
            loss_fp = ce_local(logits_fp, target_u)
            loss_u = lambda_ * loss_fp + mu * 0.5 * loss_s


            loss = 0.5 * (loss_sup + loss_u)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

        avg_loss = running_loss / (steps if steps>0 else 1)
        mean_fraction = float(np.mean(fractions_this_epoch)) if len(fractions_this_epoch)>0 else 0.0
        fractions_per_epoch.append(mean_fraction)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks,_ in val_loader_local:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_loss += sup_loss_local(model(imgs), masks.long()).item() * imgs.size(0)
        val_loss /= len(val_ds)
        scheduler_local.step(val_loss)

        _,_,val_dice,_ = compute_per_class_dice(model, val_loader_local, NUM_CLASSES, DEVICE, plot=False)
        history.append({
            "epoch": epoch,
            "val_loss": val_loss,
            "val_dice": val_dice
        })


        print(f"[Fold {fold}] Phase2 E{epoch}/{num_epochs} — train_avg:{avg_loss:.4f} | val:{val_loss:.4f} | val_dice: {val_dice:.4f} | mean_frac:{mean_fraction:.3f}")
        if val_dice>best_dice:
            best_dice = val_dice

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(model_out_dir, f'best_val_fold{fold}_phase2.pth'))

    return best_val, best_dice,  history



# ------------------ K-Fold loop ------------------
kf = KFold(n_splits=K_FOLDS, shuffle=False)
results = []
fold_i = 0

for train_idx_local, val_idx_local in kf.split(base_indices):
    fold_i += 1
    print(f"\n=== Fold {fold_i}/{K_FOLDS} ===")
    train_idxs = base_indices[train_idx_local].tolist()
    val_idxs = base_indices[val_idx_local].tolist()

    print(f"Fold {fold_i}: {len(train_idxs)} train samples, {len(val_idxs)} val samples")

    train_ds_fold = LabeledCTScanDataset(IMG_DIR, MASK_CSV, transform=weak_transform, indices=train_idxs)
    val_ds_fold = LabeledCTScanDataset(IMG_DIR, MASK_CSV, transform=base_transform, indices=val_idxs)

    set_seed(SEED + fold_i)
    model_fold = create_model()

    best_val_p1, hist_p1 = train_phase1(model_fold, train_ds_fold, val_ds_fold,
                                        epochs=NUM_EPOCHS_PHASE1, batch_size=BATCH,
                                        seed=SEED + fold_i, model_out_dir=PATH, fold=fold_i)
    print(f"[Fold {fold_i}] Phase1 best val: {best_val_p1:.4f}")

    model_fold.load_state_dict(torch.load(os.path.join(PATH, f'best_val_fold{fold_i}_phase1.pth')))

    best_val_p2, best_dice_p2, history = train_phase2(model_fold, train_ds_fold, unlabeled_ds, val_ds_fold,
                                                            num_epochs=NUM_EPOCHS_PHASE2, batch_size=BATCH,
                                                            tau=TAU, fp_drop_p=FP_DROP_P,
                                                            lambda_=LAMBDA, mu=MU,
                                                            ignore_index=IGNORE_INDEX,
                                                            model_out_dir=PATH, fold=fold_i)
    print(f"[Fold {fold_i}] Phase2 best val: {best_val_p2:.4f} | Phase2 best dice: {best_dice_p2:.4f}")

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(PATH, f'fold_{fold_i}_history.csv'), index=False)

    results.append({
        "fold": fold_i,
        "n_train": len(train_idxs),
        "n_val": len(val_idxs),
        "best_phase2_val": float(best_val_p2),
        "best_phase2_dice": float(best_dice_p2),
        #"history_file": os.path.join(PATH, f'fold_{fold_i}_history.csv')
    })



    del model_fold
    torch.cuda.empty_cache()

# save results
df_res = pd.DataFrame(results).sort_values('fold').reset_index(drop=True)
print("\nKFold summary:")
print(df_res)
df_res.to_csv(os.path.join(PATH, 'kfold_results_final.csv'), index=False)
print("K-Fold (labeled-driven) pipeline complete. Results saved to:", os.path.join(PATH, 'kfold_results_final.csv'))





# ------------------ Inference phase ------------------
LABELS_TRAIN = pd.read_csv(MASK_CSV, index_col=0, header=0).T
USE_ENSEMBLE = True   

# Single-model filename (ignored when USE_ENSEMBLE=True)
MODEL_FILENAME = 'best_val_fold1_phase2.pth'
# Ensemble filenames (ignored when USE_ENSEMBLE=False)
MODEL_FILENAMES = [os.path.join(PATH, f'best_val_fold{f}_phase2.pth') for f in range(1,5)]



# Load either a single model OR the ensemble (not both)
models = []
if USE_ENSEMBLE:
    # load ensemble models
    for p in MODEL_FILENAMES:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Ensemble model file not found: {p}")
        models.append(load_model_from_file(p))
    if len(models) == 0:
        raise RuntimeError("No ensemble models loaded.")
    for m in models:
        m.eval()
    print(f"Loaded ensemble with {len(models)} models.")
else:
    # load single model
    model_path = os.path.join(PATH, MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Single model file not found: {model_path}")
    model = load_model_from_file(model_path)
    model.eval()
    print(f"Loaded single model: {model_path}")



# Inference on test set
test_ds = TestCTScanDataset(img_dir=os.path.join(PATH, 'test-images/'), transform=base_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)


all_preds, filenames = inference_with_postprocessing(
    test_loader=test_loader,
    models=models if USE_ENSEMBLE else [model],
    #head_index = TEST_HEAD_INDEX,
    device=DEVICE,
    use_ensemble=USE_ENSEMBLE,
    use_confidence_fallback=False,
)



# all_preds : (N_test, 256*256)
df = pd.DataFrame(all_preds, columns=LABELS_TRAIN.columns).T
df.columns = filenames

output_csv = os.path.join(PATH,CSV_FILENAME)
df.to_csv(output_csv, index=True)

print("Saved CSV :", output_csv)








