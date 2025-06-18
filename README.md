# MaDoUNet: A Mamba Powered Double-UNet for Lightweight Polyp Segmentation


## 📝 Abstract  
We propose **MaDoUNet**, a lightweight dual‑encoder architecture that combines **EfficientNetB4** and **DenseNet121** to extract complementary high‑level semantic and low‑level structural features. EfficientNetB4 is enhanced with depthwise convolutional transformers to improve global context modeling, while DenseNet121 is integrated with a novel **VSS‑Mamba** module for effective temporal‑spatial feature representation. A refined decoder and a composite loss function—combining Binary Cross‑Entropy and Dice Loss—further enhance segmentation performance and address class imbalance.  

Experimental results demonstrate that MaDoUNet achieves consistent improvements across key metrics, with average gains of approximately **3–5 %** in Dice score, IoU, and precision compared to existing methods. With its strong accuracy and computational efficiency, MaDoUNet offers promising potential for real‑time clinical applications in polyp segmentation.  

## Architecture

The proposed **MaDoUNet** employs a dual-encoder structure combining **EfficientNetB4** and **DenseNet121**. It incorporates depthwise transformer blocks and a custom **VSS-Mamba** module to enhance both global and local feature representation.


<p align="center">
  <img src="media\architectureDiagram.png" alt="MaDoUNet Architecture" width="700"/>
</p>

## 📊 Results

MaDoUNet demonstrates significant improvements in segmentation performance over existing models. It consistently achieves higher Dice scores, IoU, and Precision metrics across test datasets such as Kvasir-SEG and CVC-ClinicDB.

Below is a qualitative result showcasing input image, ground truth, and predicted mask side-by-side:

<p align="center">
  <img src="media\mask_prediction_wrt_groundTrutch.jpg" alt="MaDoUNet Segmentation Output" width="700"/>
</p>

**Quantitative Evaluation:**

| Dataset         | mIoU  | DSC   | Recall | Precision | F2 Score |
|-----------------|-------|-------|--------|-----------|----------|
| **Kvasir-SEG**  | 0.8536| 0.921 | 0.9403 | 0.9025    | 0.9310   |
| **CVC-ClinicDB**| 0.9027| 0.948 | 0.9556 | 0.9422    | 0.9520   |

> 📌 *Note: These results were obtained on the Kvasir-SEG and CVC-ClinicDB datasets using a 80/20 train-validation split.*



## 🔧 How to Use

Follow the steps below to train, test, visualize, or run inference using MaDoUNet.

### 📦 1. Install Dependencies

We recommend using Python 3.10 or later.

```bash
pip install -r requirements.txt
```

> Optionally, install CPU-only PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### 📁 2. Dataset Structure

Prepare your dataset (e.g., Kvasir-SEG) in the following format:

```
dataset_root/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── masks/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

---

### 🚀 3. Train the Model

```bash
python main.py --train \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --run_instance_name your_run_name
```

All training results (model checkpoint, logs) will be saved under:

```
results/train_results/your_run_name/
```

---

### 🔍 4. Visualize Predictions

```bash
python main.py --visualize \
  --checkpoint_path results/train_results/your_run_name/madounet_best.pth \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --run_instance_name your_vis_run
```

Visualization will be saved to:

```
results/visualization_results/your_vis_run/visualization.png
```

---

### 🧪 5. Evaluate on Test Set

```bash
python main.py --test \
  --checkpoint_path results/train_results/your_run_name/madounet_best.pth \
  --image_dir /path/to/images \
  --mask_dir /path/to/masks \
  --run_instance_name test_run
```

Results are saved to:

```
results/test_results/test_run/test_metrics.csv
```

---

### 🤖 6. Run Inference on a Folder of Images

```bash
python main.py --inference \
  --checkpoint_path results/train_results/your_run_name/madounet_best.pth \
  --image_dir /path/to/input_images \
  --run_instance_name infer_run
```

Predicted masks will be saved under:

```
results/inference_results/infer_run/
```

---

### ✨ Optional Arguments

- `--input_size 256 256` — Resize input images.
- `--num_samples 5` — For visualization.
- `--threshold 0.5` — Binarization threshold for mask predictions.

---

Feel free to explore or modify the pipeline via `main.py` for different use-cases.

---

---

## 👥 Contributors

We gratefully acknowledge the following contributors:

- **Kaushal Sambanna** — Lead Researcher & Developer  
- **Sanjana Jhansi Ganji** — Researcher, Dataset Preparation, Evaluation Support  
- **Srikanth Panigrahi** — Academic Guidance
- **Routhu Srinivasa Rao** - Guide

> Feel free to [open a pull request](https://github.com/kaushal380/MaDoUNet/pulls) if you'd like to contribute!

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this software with attribution.

---

## 📬 Contact

For queries, collaborations, or feedback:

- 📧 **Email:** [kaushal.sambanna@gmail.com]  
- 🧠 **LinkedIn:** [linkedin.com/in/kaushal-sambanna](https://www.linkedin.com/in/kaushal-sambanna-92b74a360/)  
- 💻 **GitHub Issues:** [Submit an issue](https://github.com/kaushal380/MaDoUNet/issues)

---

⭐️ If you find this project useful, please consider starring the repository. It helps others discover it!
