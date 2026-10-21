# Sources and attribution

* Microsoft Qlib, Alpha158-style price/volume definitions: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py . MIT notice retained in `licenses/QLIB_LICENSE`. The original reference-file hashes are recorded separately; those reference source files are not vendored here.
* Zura Kakushadze, *101 Formulaic Alphas*, https://arxiv.org/abs/1601.00991 . Local report transcription also used **Huatai Single-Factor Testing: Large-Scale Technical Factors**. `HT101_*` identifies that transcription/convention set; do not assume exact equivalence to every Alpha101 implementation.
* **Guotai Junan: A Multi-Factor Stock-Selection Framework Based on Short-Horizon Price-Volume Features** supplied the Alpha191-style formulas. `GTJA191_*` entries record formula/page/convention details. The brokerage PDFs and their narrative text/figures are not redistributed. Mathematical formulas are implemented in this repository's interpreter; MIT does not relicense the source reports.
* Scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch and pytorch-tabnet are dependencies, installed separately under their own licenses. Model factories and compact neural architectures are included; no model binaries are distributed.

The MIT license applies to original repository code. Keep upstream notices for Qlib-derived portions. Vendor datasets and source reports remain subject to their original terms.
