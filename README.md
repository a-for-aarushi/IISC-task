Overview

This notebook implements a Vision Transformer (ViT) from scratch on the CIFAR-10 dataset (10 classes) using PyTorch, following the paper:

“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” (Dosovitskiy et al., ICLR 2021)

The implementation includes:

Image patchification and embedding
Learnable positional embeddings
CLS token for classification
Transformer encoder blocks (MHSA + MLP + residual connections + normalization)
Classification head based on the CLS token output

Best performance setup
| Parameter                | Value            |
| ------------------------ | ---------------- |
| Patch Size               | 4×4              |
| Embedding Dimension      | 256              |
| Number of Heads          | 8                |
| Number of Encoder Blocks | 6                |
| MLP Hidden Dim           | 512              |
| Dropout                  | 0.1              |
| Optimizer                | AdamW            |
| Learning Rate            | 3e-4             |
| Scheduler                | Cosine Annealing |
| Batch Size               | 128              |
| Epochs                   | 100              |

nalysis

Effect of Patch Size

Smaller patches (4×4) capture finer details and improve accuracy but increase training time.
Larger patches (8×8 or 16×16) reduce computation but lose local information.

Depth/Width Trade-offs
Increasing depth beyond 6 caused overfitting due to limited dataset size.
Wider embeddings improved stability and faster convergence.

Augmentation and Optimization
RandomCrop and horizontal flip improved generalization by ~2%.
AdamW optimizer provided better regularization than Adam.

Limitations

Training ViT from scratch on CIFAR-10 requires high computation time.
Small datasets limit transformer efficiency.
Pretraining or hybrid CNN-ViT models (e.g., DeiT) can enhance performance.


Q2 — Text-Driven Image Segmentation with SAM 2
Overview

This notebook performs text-prompted image segmentation using Segment Anything Model 2 (SAM 2).
It combines natural language understanding with image segmentation by linking text-based region localization and mask generation.

Pipeline

Load Image: Upload or specify an image URL.

Text Prompt Input: Enter a text prompt (e.g., "a dog", "the blue car").

Text-to-Region Conversion:

Uses GroundingDINO, CLIPSeg, or GLIP to identify regions relevant to the text.

Produces bounding boxes or coarse masks as region seeds.

Segmentation with SAM 2:

Seeds are refined into accurate masks using SAM 2.

Visualization:

The final mask is overlaid on the image for clarity.
