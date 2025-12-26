"""
下载 SAM (Segment Anything Model) 权重文件

支持下载:
- ViT-H (sam_vit_h_4b8939.pth) - 2.56GB, 最高精度
- ViT-L (sam_vit_l_0b3195.pth) - 1.25GB, 平衡性能
- ViT-B (sam_vit_b_01ec64.pth) - 375MB, 最快速度
"""

import os
import urllib.request
from pathlib import Path


# SAM 模型下载链接
SAM_MODELS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "file": "sam_vit_h_4b8939.pth",
        "size_gb": 2.56,
        "description": "ViT-H - 最高精度 (推荐)"
    },
    "vit_l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "file": "sam_vit_l_0b3195.pth",
        "size_gb": 1.25,
        "description": "ViT-L - 平衡性能"
    },
    "vit_b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "file": "sam_vit_b_01ec64.pth",
        "size_gb": 0.375,
        "description": "ViT-B - 最快速度"
    }
}


def download_with_progress(url: str, dest_path: str):
    """带进度条的下载函数"""
    print(f"正在下载: {url}")
    print(f"保存到: {dest_path}")

    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        mb_downloaded = count * block_size / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r进度: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print("\n下载完成!")
        return True
    except Exception as e:
        print(f"\n下载失败: {e}")
        return False


def main():
    # 确保 weights 目录存在
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    print("=== SAM 模型下载脚本 ===\n")
    print("可用的模型版本:")
    for key, info in SAM_MODELS.items():
        print(f"  [{key}] {info['description']} - {info['size_gb']:.2f}GB")

    # 选择模型
    choice = input("\n请选择要下载的模型 (vit_h/vit_l/vit_b) [默认: vit_h]: ").strip()
    if not choice:
        choice = "vit_h"

    if choice not in SAM_MODELS:
        print(f"无效的选择: {choice}")
        return

    model_info = SAM_MODELS[choice]
    dest_path = weights_dir / model_info["file"]

    # 检查是否已存在
    if dest_path.exists():
        overwrite = input(f"文件 {dest_path} 已存在，是否覆盖? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("取消下载")
            return

    # 下载
    success = download_with_progress(model_info["url"], str(dest_path))

    if success:
        # 更新配置
        print(f"\n模型已保存到: {dest_path}")
        print(f"\n请更新你的配置:")
        if choice == "vit_h":
            print("  sam_checkpoint = \"weights/sam_vit_h_4b8939.pth\"")
            print("  sam_model_type = \"vit_h\"")
        elif choice == "vit_l":
            print("  sam_checkpoint = \"weights/sam_vit_l_0b3195.pth\"")
            print("  sam_model_type = \"vit_l\"")
        else:
            print("  sam_checkpoint = \"weights/sam_vit_b_01ec64.pth\"")
            print("  sam_model_type = \"vit_b\"")


if __name__ == "__main__":
    main()
