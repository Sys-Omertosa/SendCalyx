# Third-party notices

SendCalyx itself is released under the [MIT License](LICENSE). The components it
builds on, and the data its model checkpoints derive from, carry their own terms.

## Software

| Component | License |
| --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch) | BSD 3-Clause |
| [torchvision](https://github.com/pytorch/vision) | BSD 3-Clause |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache 2.0 |
| [FastAPI](https://github.com/fastapi/fastapi) | MIT |
| [Uvicorn](https://github.com/encode/uvicorn) | BSD 3-Clause |
| [Pydantic](https://github.com/pydantic/pydantic) | MIT |
| [Pillow](https://github.com/python-pillow/Pillow) | MIT-CMU |
| [NumPy](https://github.com/numpy/numpy) | BSD 3-Clause |
| [OpenCV](https://github.com/opencv/opencv-python) | Apache 2.0 |
| [Matplotlib](https://github.com/matplotlib/matplotlib) | PSF-based |
| [React](https://github.com/facebook/react) | MIT |
| [Vite](https://github.com/vitejs/vite) | MIT |
| [Tailwind CSS](https://github.com/tailwindlabs/tailwindcss) | MIT |
| [Motion](https://github.com/motiondivision/motion) | MIT |
| [Lucide](https://github.com/lucide-icons/lucide) | ISC |
| [Axios](https://github.com/axios/axios) | MIT |

This list covers the direct dependencies. Each carries its own transitive
dependencies under their respective licenses.

## Fonts

The wordmark is set in **Climax** by SelarasWP. The font file is not covered by
this repository's MIT license, and its terms for web embedding and
redistribution have not been established. Confirm licensing with the foundry
before deploying publicly. The stack falls back to
[Archivo](https://fonts.google.com/specimen/Archivo) (SIL Open Font License 1.1)
where Climax is unavailable. Interface text uses
[Inter Tight](https://fonts.google.com/specimen/Inter+Tight) and
[JetBrains Mono](https://fonts.google.com/specimen/JetBrains+Mono), both under
the SIL Open Font License 1.1.

## Pretrained backbone weights

InceptionV3 and InceptionResNetV2 are initialised from ImageNet-pretrained
weights distributed with torchvision and `timm`, under those projects' terms.

## Model checkpoints and training data

The `.pth` checkpoints in `backend/models/` are derived from the components above
and from medical imaging data whose redistribution terms are set by the original
dataset providers, not by this repository's license. The training data itself is
not distributed here. See the dataset sources section of the
[README](README.md) for provenance.

Software licensing does not grant medical imaging licensing. Anyone
redistributing the checkpoints or reproducing training should verify the terms of
the underlying datasets independently.
