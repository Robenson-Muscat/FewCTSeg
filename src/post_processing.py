
import numpy as np
import cv2
from collections import Counter



def relabel_fp_by_convex_hull(pred_mask,
                              gt_mask=None,
                              min_area_for_isolated=20,
                              ignore_label=0,
                              overlap_threshold_with_gt=0,
                              use_gt_labels_inside_hull=False,
                              verbose=False):
    """
    Relabel isolated false-positive connected components in `pred_mask`
    by assigning them the label of the largest label present inside their convex hull.
    - pred_mask: 2D numpy array (H,W) with integer labels (0 = background).
    - gt_mask: optional 2D numpy array (H,W) containing true labels (same encoding).
               If provided, a predicted component is considered FP if it has
               <= overlap_threshold_with_gt pixels overlapping with the same label in gt_mask.
               If None, components with area <= min_area_for_isolated are considered "isolated".
    - min_area_for_isolated: used only when gt_mask is None (tunable).
    - ignore_label: label to treat as background (default 0). We never choose 0 as replacement.
    - overlap_threshold_with_gt: minimal number of overlapping pixels with same label in gt to consider it NOT FP.
    - use_gt_labels_inside_hull: if True and gt_mask is provided, choose replacement label from gt_mask inside hull
                                 (preferred if you want GT-based relabel).
    - verbose: prints basic stats.
    Returns: a copy of pred_mask with relabelings applied.
    """
    H, W = pred_mask.shape
    out = pred_mask.copy()
    unique_labels = np.unique(pred_mask)
    stats = {"checked":0, "relabelled":0, "skipped_no_candidate":0}

    for lab in unique_labels:
        if lab == ignore_label:
            continue
        # binary mask for current predicted label
        binary = (pred_mask == lab).astype(np.uint8)
        if binary.sum() == 0:
            continue
        # connected components on this class mask
        ncomp, comp_lbl, comp_stats, comp_centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        # iterate components (skip background comp 0)
        for cid in range(1, ncomp):
            comp_mask = (comp_lbl == cid)
            comp_area = int(comp_stats[cid, cv2.CC_STAT_AREA])
            stats["checked"] += 1

            # decide whether this component is considered FP / isolated
            is_fp = False
            if gt_mask is not None:
                # number of pixels overlapping with same label in GT
                overlap_same = int(np.logical_and(comp_mask, gt_mask == lab).sum())
                if overlap_same <= overlap_threshold_with_gt:
                    is_fp = True
            else:
                # if no GT, treat small components as isolated
                if comp_area <= min_area_for_isolated:
                    is_fp = True

            if not is_fp:
                continue

            # get contours for this component
            comp_mask_uint8 = (comp_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(comp_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            # merge all contour points (rarely more than one)
            all_pts = np.vstack(contours).squeeze()
            if all_pts.ndim == 1:
                # single point component -> no sensible hull
                stats["skipped_no_candidate"] += 1
                continue
            hull = cv2.convexHull(all_pts)
            hull_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(hull_mask, hull, 1)

            # candidate label sources inside hull (exclude the component pixels themselves)
            hull_excluding_comp = np.logical_and(hull_mask.astype(bool), ~comp_mask)

            # choose source for candidate labels
            if use_gt_labels_inside_hull and (gt_mask is not None):
                labels_in_hull = gt_mask[hull_excluding_comp]
            else:
                labels_in_hull = pred_mask[hull_excluding_comp]

            if labels_in_hull.size == 0:
                stats["skipped_no_candidate"] += 1
                continue

            # count occurrences, ignore background (ignore_label)
            counts = Counter(labels_in_hull.tolist())
            if ignore_label in counts:
                del counts[ignore_label]
            if len(counts) == 0:
                # nothing but background inside hull
                stats["skipped_no_candidate"] += 1
                continue

            # pick the most common label in hull
            chosen_label, chosen_count = counts.most_common(1)[0]

            # Apply relabelling
            out[comp_mask] = chosen_label
            stats["relabelled"] += 1

            if verbose:
                print(f"Comp class {lab} area={comp_area} -> relabelled to {chosen_label} (count in hull={chosen_count})")

    if verbose:
        print(f"Relabel FP: checked {stats['checked']} comps, relabelled {stats['relabelled']}, skipped {stats['skipped_no_candidate']}")
    return out


def inference_with_postprocessing(
    test_loader,
    models,
    head_index,
    device,
    use_ensemble=True,
    min_area_for_isolated=20,
    use_confidence_fallback=True,
    threshold=0.95
):
    """
    Runs inference on test_loader with optional ensemble and convex-hull post-processing.
    Returns:
        all_preds: numpy array (N_test, H*W)
        filenames: list of filenames
    """
    all_preds = []
    filenames = []

    test_head_set = set(head_index.tolist()) 


    with torch.no_grad():
        for imgs, names, indices in tqdm(test_loader, desc="Inference + postproc"):
            imgs = imgs.to(device)

            # ---------- forward ----------
            if use_ensemble:
                logits_acc = None
                for m in models:
                    logits_m = m(imgs)  # (B,C,H,W)
                    logits_acc = logits_m if logits_acc is None else logits_acc + logits_m
                logits_acc /= len(models)
                preds = torch.argmax(logits_acc, dim=1)

                #  softmax
                probs = torch.softmax(logits_acc, dim=1)

                #  top-2
                top2_probs, top2_classes = torch.topk(probs, k=2, dim=1)

                conf_map = top2_probs[:, 0, :, :]
                preds    = top2_classes[:, 0, :, :]

                second_conf  = top2_probs[:, 1, :, :]
                second_class = top2_classes[:, 1, :, :]



                if use_confidence_fallback:

                    condition = (((preds == 21)  | (preds == 24))  & (conf_map < 0.80))
                    preds= torch.where(condition, second_class, preds)

                    # Apply only for test images whose index is in TEST_HEAD_INDEX
                    condition2 = (preds == 0) & (conf_map < threshold)

                    #
                    mask_head = torch.tensor(
                        [idx.item() in test_head_set for idx in indices],
                        device=preds.device
                    )
                    mask_head = mask_head[:, None, None]

                    preds = torch.where(condition2 & mask_head, second_class, preds)



            else:
                preds = torch.argmax(models[0](imgs), dim=1)

            preds_np = preds.cpu().numpy()  # (B,H,W)



            # ---------- post-processing ----------

            for i in range(preds_np.shape[0]):
                p = preds_np[i]

                p_pp = relabel_fp_by_convex_hull(
                    p,
                    gt_mask=None,
                    min_area_for_isolated=min_area_for_isolated,
                    verbose=False
                )

                all_preds.append(p_pp.flatten())
                filenames.append(names[i])

    return np.stack(all_preds, axis=0), filenames




def inference_with_postprocessing(
    test_loader,
    models,
    head_index,
    device,
    use_ensemble=True,
    min_area_for_isolated=20,
    use_confidence_fallback=True,
    threshold=0.95
):
    """
    Runs inference on test_loader with optional ensemble and convex-hull post-processing.
    Returns:
        all_preds: numpy array (N_test, H*W)
        filenames: list of filenames
    """
    all_preds = []
    filenames = []

    # set pour lookup rapide
    test_head_set = set(head_index.tolist())

    with torch.no_grad():
        for imgs, names, indices in tqdm(test_loader, desc="Inference + postproc"):
            imgs = imgs.to(device)

            # ---------- forward ----------
            if use_ensemble:
                logits_acc = None
                for m in models:
                    logits_m = m(imgs)  # (B,C,H,W)
                    logits_acc = logits_m if logits_acc is None else logits_acc + logits_m
                logits_acc /= len(models)

                # softmax + top2
                probs = torch.softmax(logits_acc, dim=1)
                top2_probs, top2_classes = torch.topk(probs, k=2, dim=1)

                conf_map = top2_probs[:, 0, :, :]
                preds    = top2_classes[:, 0, :, :]

                second_conf  = top2_probs[:, 1, :, :]
                second_class = top2_classes[:, 1, :, :]

                if use_confidence_fallback:

                    # -------- condition 1 --------
                    condition = (((preds == 21) | (preds == 24)) & (conf_map < 0.80))
                    preds = torch.where(condition, second_class, preds)

                    # -------- condition 2 --------
                    condition2 = (preds == 0) & (conf_map < threshold)

                    #  Apply only for test images whose index is in TEST_HEAD_INDEX 
                    mask_head = torch.tensor(
                        [idx.item() in test_head_set for idx in indices],
                        device=preds.device)

                    mask_head = mask_head[:, None, None]
                    preds = torch.where(condition2 & mask_head, second_class, preds)

            else:
                preds = torch.argmax(models[0](imgs), dim=1)

            preds_np = preds.cpu().numpy()  # (B,H,W)

            # ---------- post-processing ----------
            for i in range(preds_np.shape[0]):
                p = preds_np[i]

                p_pp = relabel_fp_by_convex_hull(
                    p,
                    gt_mask=None,
                    min_area_for_isolated=min_area_for_isolated,
                    verbose=False
                )

                all_preds.append(p_pp.flatten())
                filenames.append(names[i])

    return np.stack(all_preds, axis=0), filenames









#-----------Post-processing en insérant de la field knowledge--------

# logits (C,H,W) for a single image, forbidden_classes indices
#logits[forbidden_classes, :, :] = -1e9   # efface toute probabilité de ces classes
#pred = logits.argmax(dim=0)

