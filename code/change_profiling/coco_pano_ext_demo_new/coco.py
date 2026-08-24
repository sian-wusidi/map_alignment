import numpy as np
import pandas as pd
from . import iou
from .prec_recall_map import colorize_regions



def COCO(A: np.ndarray, B: np.ndarray, mode=None, ignore_zero=True, output_scores = False, pairing_threshold=0.5):
    """
    
    Compute the COCO Panoptic metric in a bipartite graph (A ↔ B) with degree 1.

    ![](figs/graphs.svg)

    Three modes can be used:

    * Segmentation: A and B are supposed to be two 2D binary images. The labelisation of the connected components is then performed and
      the IoU between each pair of components is computed.
    * Labelmaps: same as previously but the the images are already labeled
    * IoU Pairing Arrays: A and B are 1D arrays that represents the matching weights in a bipartite graph (A ↔ B) where a component
      in A is in relation with at most one component in B.


    Args:
        A (np.ndarray): Binary image, labelmap, or 1D weights array
        B (np.ndarray): Binary image, labelmap, or 1D weights array
        mode (string, optional): "segmentation", "labelmap", "iou_array". Defaults to None which tries to deduce the
                                  mode from inputs.
        ignore_zero (bool, optional): Ignore the label (node) 0 which is usually the background. Defaults to True.
        output_scores (bool, optional): Output the dataframe with IoU and F-Scores as the last item of the tuple returned.
        iou_threshold (float, optional): In [0.5 - 1] The minimum overlap rate (IoU) that components should get to be paired
    Raises:
        ValueError: If mode is invalid or unable to deduce
    
    Returns:
        tuple: The triplet (Panoptic Quality, Segmentation Quality, Recognition Quality, Scores_Dataframe)
    """    
    A = np.asarray(A)
    B = np.asarray(B)

    if mode not in {None, "segmentation", "labelmap", "iou_array"}:
        raise ValueError(f"Invalid mode '{mode}'.")

    if not (0 < pairing_threshold < 1):
        raise ValueError(f"Invalid IoU threshold '{pairing_threshold}'.")

    if not mode:
        modeA, modeB = _deduce_mode(A, B)
    else:
        modeA = modeB = mode


    if modeA == "segmentation": A = _compute_labelmap(A)
    if modeB == "segmentation": B = _compute_labelmap(B)

    if modeA != "iou_array" and modeB != "iou_array":
        A, B = _compute_iou(A, B)

    if ignore_zero:
        A = A[1:]
        B = B[1:]

    #print(f"Number of labels in A: {A.size}")
    #print(f"Number of labels in B: {B.size}")

    if B.size == 0:
        print("Warning: empty prediction. Setting scores to 0 and skipping plot generation.")
        return 0., 0., 0.

    df = iou.compute_matching_scores(A, B, pairing_threshold)
    #if plot:
    #    iou.plot_scores(df, out=plot)

    COCO_SQ = df["IoU"].mean() if len(df) > 0 else 0
    COCO_RQ = df["F-score"].iloc[0] if len(df) > 0 else 0
    COCO_PQ = COCO_SQ * COCO_RQ
    if not output_scores:
        df = None 
    import pdb
    pdb.set_trace() 
    return COCO_PQ, COCO_SQ, COCO_RQ, df

def AP(A: np.ndarray, B: np.ndarray, mode=None, ignore_zero=True, pairing_threshold=0.5):
    """
    
    Compute the COCO Panoptic metric in a bipartite graph (A ↔ B) with degree 1.

    ![](figs/graphs.svg)

    Three modes can be used:

    * Segmentation: A and B are supposed to be two 2D binary images. The labelisation of the connected components is then performed and
      the IoU between each pair of components is computed.
    * Labelmaps: same as previously but the the images are already labeled
    * IoU Pairing Arrays: A and B are 1D arrays that represents the matching weights in a bipartite graph (A ↔ B) where a component
      in A is in relation with at most one component in B.


    Args:
        A (np.ndarray): Binary image, labelmap, or 1D weights array
        B (np.ndarray): Binary image, labelmap, or 1D weights array
        mode (string, optional): "segmentation", "labelmap", "iou_array". Defaults to None which tries to deduce the
                                  mode from inputs.
        ignore_zero (bool, optional): Ignore the label (node) 0 which is usually the background. Defaults to True.
        iou_threshold (float, optional): In [0.5 - 1] The minimum overlap rate (IoU) that components should get to be paired
    Raises:
        ValueError: If mode is invalid or unable to deduce
    
    Returns:
        tuple: The triplet (Panoptic Quality, Segmentation Quality, Recognition Quality, Scores_Dataframe)
    """    
    A = np.asarray(A)
    B = np.asarray(B)

    if mode not in {None, "segmentation", "labelmap", "iou_array"}:
        raise ValueError(f"Invalid mode '{mode}'.")

    if not (0 < pairing_threshold < 1):
        raise ValueError(f"Invalid IoU threshold '{pairing_threshold}'.")

    if not mode:
        modeA, modeB = _deduce_mode(A, B)
    else:
        modeA = modeB = mode


    if modeA == "segmentation": A = _compute_labelmap(A)
    if modeB == "segmentation": B = _compute_labelmap(B)

    if modeA != "iou_array" and modeB != "iou_array":
        A, B = _compute_iou(A, B)

    if ignore_zero:
        A = A[1:]
        B = B[1:]

    TP = (A >= pairing_threshold)
    FP = (A < pairing_threshold)
    acc_FP = np.cumsum(FP)
    acc_TP = np.cumsum(TP)
    rec = acc_TP/len(B)  # ground truth labels
    prec = np.divide(acc_TP, (acc_FP + acc_TP))
    
    AP_score, mpre, mrec, _ = CalculateAveragePrecision(rec, prec)

    if B.size == 0:
        print("Warning: empty prediction. Setting scores to 0 and skipping plot generation.")
        return 0., 0., 0.


    return AP_score, np.sum(TP), np.sum(FP)


def COCO_plot(df: pd.DataFrame, ax=None, lower_bound = 0.5):
    """[summary]

    Args:
        df (pd.DataFrame): The dataframe returned by the COCO function
        ax (Matplotlib.Axes, optional): Optional axis where to draw the figure. Defaults to None.
    """
    if df.empty:
        raise ValueError("Empty dataframe !")

    # sns.set()
    df = df[
        ["IoU", "Precision", "Recall", "F-score"]
    ]  # , "COCO_PQ", "COCO_SQ", "COCO_RQ"]]
    df = df.set_index("IoU")
    df = df.reindex([0] + list(df.index) + [1], method="backfill")
    df.iloc[-1] = [0, 0, 0]

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))

    df.plot(ax=ax, marker="o", drawstyle="steps-pre")
    ax.set_xlim(lower_bound, 1)
    ax.set_ylim(0, 1)


def CalculateAveragePrecision(rec, prec):
    """"
    Calculate the average precision of a precision-recall curve
    https://github.com/rafaelpadilla/Object-Detection-Metrics/"""
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1+i] != mrec[i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def precision_recall_maps(groundtruth: np.ndarray, contender: np.ndarray, mode=None, ignore_zero=True, lower_bound=0.5):
    """[summary]

    Args:
        groundtruth (np.ndarray): The groundtruth segmentation or labelmap
        contender (np.ndarray): The segmentation or labelmap to evaluate
        mode (str, optional): None, "segmentation" or "labelmap". Defaults to None and the mode is deduced from image types.
        ignore_zero (bool, optional): [description]. Defaults to True.
        lower_bound (float, optional): Must be in [0,1]. The center of the colormap (threshold of acceptance)
    """

    A = np.asarray(groundtruth)
    B = np.asarray(contender)

    if mode not in {None, "segmentation", "labelmap"}:
        raise ValueError(f"Invalid mode '{mode}'.")

    if not mode:
        modeA, modeB = _deduce_mode(A, B)
        assert modeA in {"segmentation", "labelmap"}
        assert modeB in {"segmentation", "labelmap"}
    else:
        modeA = modeB = mode


    if modeA == "segmentation": A = _compute_labelmap(A)
    if modeB == "segmentation": B = _compute_labelmap(B)

    wA, wB = _compute_iou(A, B)
    recall = colorize_regions(A, wA, lower_bound)
    precision = colorize_regions(B, wB, lower_bound)
    return precision, recall


def precision_recall_only(groundtruth: np.ndarray, contender: np.ndarray, mode=None, ignore_zero=True, lower_bound=0.5):
    """[summary]

    Args:
        groundtruth (np.ndarray): The groundtruth segmentation or labelmap
        contender (np.ndarray): The segmentation or labelmap to evaluate
        mode (str, optional): None, "segmentation" or "labelmap". Defaults to None and the mode is deduced from image types.
        ignore_zero (bool, optional): [description]. Defaults to True.
        lower_bound (float, optional): Must be in [0,1]. The center of the colormap (threshold of acceptance)
    """

    A = np.asarray(groundtruth)
    B = np.asarray(contender)

    if mode not in {None, "segmentation", "labelmap"}:
        raise ValueError(f"Invalid mode '{mode}'.")

    if not mode:
        modeA, modeB = _deduce_mode(A, B)
        assert modeA in {"segmentation", "labelmap"}
        assert modeB in {"segmentation", "labelmap"}
    else:
        modeA = modeB = mode


    if modeA == "segmentation": A = _compute_labelmap(A) #recall
    if modeB == "segmentation": B = _compute_labelmap(B) #precision

    # wA, wB, tar_cA, tar_cB, cA, cB, match_A, match_B  = _compute_iou(A, B)
    wA, wB = _compute_iou(A, B)
    wA[0] = -1 # remove roads
    wB[0] = -1 # remove roads
    precision = wB[B]
    recall = wA[A]
    # tar_cA = tar_cA[A]
    # tar_cB = tar_cB[B]
    # cA = cA[A]
    # cB = cB[B]
    #indices_A = match_A[A]
    #indices_B = match_B[B]

    return precision, recall

def _deduce_mode_1(A):
    label_types = (np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64)
    if A.ndim == 2:
        if A.dtype == "bool":
            return "segmentation"
        elif A.dtype in label_types:
            return "labelmap"
        elif A.dtype == "uint8":
            a_label_count = np.unique(A)
            if len(a_label_count) > 2:
                return "labelmap"
            else:
                return "segmentation"
    if A.ndim == 1:
        return "iou_array"
    return None

def _deduce_mode(A, B):
    mode_A = _deduce_mode_1(A)
    mode_B = _deduce_mode_1(B)

    if not mode_A:
        raise ValueError(f"Unable to deduce computation mode for COCO metric ({A.ndim}:{A.dtype})")
    if not mode_B:
        raise ValueError(f"Unable to deduce computation mode for COCO metric ({B.ndim}:{B.dtype})")


    if (mode_A == "iou_array") != (mode_B == "iou_array"):
        raise ValueError(f"Incompatible deduced mode for COCO metric: {mode_A} vs {mode_B}")

    return mode_A, mode_B



def _compute_labelmap(A):
    from skimage.measure import label as labelize
    A = labelize(A.astype(np.uint8), connectivity=1)  # , ltype=cv2.CV_16U
    return A


def _compute_iou(A, B):
    hist_inter_2d = iou.intersections(A, B)
    iou_a, iou_b, best_a, best_b = iou.compute_bipartite_edge_weigths(hist_inter_2d, mode="jaccard")
    tar_cov_a, tar_cov_b, _, _  = iou.compute_bipartite_edge_weigths(hist_inter_2d, mode="target_coverage")
    cov_a, cov_b,_,_  = iou.compute_bipartite_edge_weigths(hist_inter_2d, mode="coverage")
    return iou_a, iou_b #, tar_cov_a, tar_cov_b, cov_a, cov_b, best_a, best_b