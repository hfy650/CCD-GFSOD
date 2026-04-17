import os
import cv2
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer


def build_predictor(cfg_path, weight_path, score_thresh=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.DEVICE = "cuda"
    return DefaultPredictor(cfg)


def visualize_dataset(predictor, dataset_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)

    for data in tqdm(dataset_dicts):
        img_path = data["file_name"]
        img_name = os.path.basename(img_path)

        img = cv2.imread(img_path)
        outputs = predictor(img)

        visualizer = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=1.0
        )
        vis_img = visualizer.draw_instance_predictions(
            outputs["instances"].to("cpu")
        ).get_image()[:, :, ::-1]

        cv2.imwrite(os.path.join(save_dir, img_name), vis_img)


if __name__ == "__main__":

    # =========================
    # 1. 数据集
    # =========================
    DATASET_NAME = "voc_2007_test"

    # =========================
    # 2. 配置文件（你的命名方式完全OK）
    # =========================
    CFG_PATH = "configs/voc/defrcn_gfsod_r101_novel1_1shot_seed0.yaml"

    # =========================
    # 3. 权重路径（按你自己的改）
    # =========================
    WEIGHT_BASELINE = "checkpoints/.../baseline/model_final.pth"
    WEIGHT_CFCM = "checkpoints/.../cfcm/model_final.pth"
    WEIGHT_FULL = "checkpoints/.../full/model_final.pth"

    # =========================
    # 4. 输出目录
    # =========================
    SAVE_ROOT = "voc_vis_results"
    SAVE_BASE = os.path.join(SAVE_ROOT, "baseline")
    SAVE_CFCM = os.path.join(SAVE_ROOT, "cfcm")
    SAVE_FULL = os.path.join(SAVE_ROOT, "full")

    # =========================
    # 5. 构建预测器
    # =========================
    predictor_base = build_predictor(CFG_PATH, WEIGHT_BASELINE)
    predictor_cfcm = build_predictor(CFG_PATH, WEIGHT_CFCM)
    predictor_full = build_predictor(CFG_PATH, WEIGHT_FULL)

    # =========================
    # 6. 开始跑
    # =========================
    print("Visualizing baseline...")
    visualize_dataset(predictor_base, DATASET_NAME, SAVE_BASE)

    print("Visualizing CFCM...")
    visualize_dataset(predictor_cfcm, DATASET_NAME, SAVE_CFCM)

    print("Visualizing FULL model...")
    visualize_dataset(predictor_full, DATASET_NAME, SAVE_FULL)

    print("Done.")
