import torch
import cv2
import numpy as np
import glob
import os
import sys


def setup_config(
    config_file="config/classification/food_image/ehfr_net_food101.yaml",
    checkpoint_path="food101_results/ehfr_net_width_multiplier_1.0/checkpoint_ema_best.pt",
    width_multiplier=1.0,
    n_classes=101,
):
    """
    设置配置和加载模型
    """
    # 1. 添加命令行参数
    sys.argv.append("--common.config-file")
    sys.argv.append(config_file)

    sys.argv.append("--common.override-kwargs")
    sys.argv.append(f"model.classification.ehfr_net.width_multiplier={width_multiplier}")
    sys.argv.append(f"model.classification.n_classes={n_classes}")

    sys.argv.append("--model.classification.pretrained")
    sys.argv.append(checkpoint_path)


def visualize_last_layer_gate(
    image_path="./cam_relative_file/food101/origin/*.jpg",
    save_path="./cam_relative_file/food101/ablation_1/gate_results",
    size=(256, 256),
):
    """
    可视化最后一层 HBlock 的 Gate 值

    Args:
        image_path: 原始图片路径
        save_path: 结果保存路径
        size: 输入图片大小
    """
    from pytorch_grad_cam.utils.image import preprocess_image
    from options.opts import get_training_arguments
    from cvnets import get_model

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 1. 加载模型
    opts = get_training_arguments()
    model = get_model(opts)
    model.eval()

    # 2. 遍历所有图片
    for file_name in glob.glob(image_path):
        print(f"\n处理图片: {file_name}")

        # 读取图片
        rgb_img = cv2.imread(file_name, 1)[:, :, ::-1]  # BGR -> RGB
        if size is not None:
            rgb_img = cv2.resize(rgb_img, size)
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(
            rgb_img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # 3. 前向传播
        with torch.no_grad():
            _ = model(input_tensor)

        # 4. 获取最后一层 HBlock 的 gate
        last_hblock = model.layer_5[-1]
        gate = last_hblock.current_gate

        if gate is None:
            print("⚠️  未获取到 gate！请检查：")
            print("   1. hblock.py 中门控融合的代码是否已取消注释")
            print("   2. 第 365 行的 self.current_gate = gate 是否取消注释")
            continue

        # 5. 处理 gate 的统计信息
        gate_np = gate.squeeze().cpu().numpy()
        print(f"   Gate 统计: mean={gate_np.mean():.4f}, min={gate_np.min():.4f}, max={gate_np.max():.4f}")

        # 6. 可视化 gate

        # 如果 gate 是 (C, H, W)，取平均
        if gate_np.ndim == 3:
            gate_np = gate_np.mean(axis=0)

        # 归一化到 [0, 1]
        gate_np = (gate_np - gate_np.min()) / (gate_np.max() - gate_np.min() + 1e-8)

        # 转成彩色热力图 (JET)
        gate_colored = cv2.applyColorMap(np.uint8(255 * gate_np), cv2.COLORMAP_JET)
        
        # 把 gate 热力图 resize 到和原图一样大
        img_h, img_w = rgb_img.shape[:2]
        gate_colored = cv2.resize(gate_colored, (img_w, img_h))

        # 叠加到原图
        overlay = cv2.addWeighted(
            np.uint8(255 * rgb_img[:, :, ::-1]), 0.5, gate_colored, 0.5, 0
        )

        # 保存结果
        base_name = os.path.basename(file_name)
        cv2.imwrite(os.path.join(save_path, f"gate_{base_name}"), gate_colored)
        cv2.imwrite(os.path.join(save_path, f"overlay_{base_name}"), overlay)
        print(f"   保存到: {save_path}")

        # 7. 还可以保存 1-gate（即关注全局特征的部分
        gate_global = 1 - gate_np
        gate_global = (gate_global - gate_global.min()) / (gate_global.max() - gate_global.min() + 1e-8)
        gate_global_colored = cv2.applyColorMap(np.uint8(255 * gate_global), cv2.COLORMAP_JET)
        gate_global_colored = cv2.resize(gate_global_colored, (img_w, img_h))
        cv2.imwrite(os.path.join(save_path, f"gate_global_{base_name}"), gate_global_colored)


if __name__ == "__main__":
    # 设置参数
    setup_config(
        config_file="config/classification/food_image/ehfr_net_food101.yaml",
        checkpoint_path="food101_results/ehfr_net_width_multiplier_1/checkpoint_ema_best.pt",
        width_multiplier=1.0,
        n_classes=101,
    )

    # 运行可视化
    visualize_last_layer_gate(
        image_path="./cam_relative_file/food101/origin/*.jpg",
        save_path="./cam_relative_file/food101/gated_ehfr_net/gate_results",
        size=(256, 256),
    )
