<div align="center">
	<h1>Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning</h1>
</div>

<div align="center">
    <p>
        <a href="https://www.zhongxiaocong.com/">Zhongxiao Cong</a> &nbsp;&nbsp;
        <a href="https://qitaozhao.github.io/">Qitao Zhao</a>&nbsp;&nbsp;
        <a href="https://msjeon.me/">Minsik Jeon</a>&nbsp;&nbsp;
        <a href="https://shubhtuls.github.io/">Shubham Tulsiani</a>
    </p>
    <p>
        Carnegie Mellon University
    </p>
</div>


<div align="center">
	<a href="https://arxiv.org/abs/2602.20157"><img src="https://img.shields.io/badge/arXiv-2512.10950-b31b1b" alt="arXiv"></a>
	<a href="https://flow3r-project.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
	<!-- <a href=""><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> -->
</div>

<div align="center">
    <img src="assets/teaser_video.gif" alt="overview">
</div>
	
<!-- ![teaser](https://raw.githubusercontent.com/QitaoZhao/QitaoZhao.github.io/main/research/E-RayZer/images/erayzer_teaser.png) -->

## Overview
Flow3r augments visual geometry learning with dense 2D correspondences (`flow') as supervision, enabling scalable training from unlabeled monocular videos. Flow3r achieves state-of-the-art results across eight benchmarks spanning static and dynamic scenes, with its largest gains on in-the-wild dynamic videos where labeled data is most scarce.

            

## Quick Start

### 1. Create the environment
```bash
conda create -n flow3r python=3.11
conda activate flow3r

pip install -r requirements.txt
```
### 2. Download and place checkpoint
- `flow3r.bin`: Flow3r trained on ~834k video sequences.

Please fetch the checkpoint manually from [Google Drive](https://drive.google.com/drive/folders/1BYkkpf8L8QMa3zhLG7ACnIN3jgtXbT3f?usp=sharing) and drop the file into `checkpoints/`.

### 3. Launch the Gradio app
```bash
python gradio_app.py 
```

<!-- 
## License
- **Code**: MIT License (see `LICENSE`).
- **Model weights**: Adobe Research License (see `LICENSE-WEIGHTS`).  The model weights are **not** covered by the MIT License. -->

## Acknowledgements
- Our work builds upon several fantastic open-source projects. We would like to acknowledge and thank the authors of:
    - [Pi3](https://github.com/yyfz/Pi3?tab=readme-ov-file)
    - [VGGT](https://github.com/facebookresearch/vggt)
- We also thank the members of the [Physical Perception Lab](https://shubhtuls.github.io/) at CMU for their valuable discussions.

## Citation
If you find our work useful, please cite:
```bibtex
@inproceedings{cong2026flow3r,
    title={Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning},
    author={Cong, Zhongxiao and Zhao, Qitao and Jeon, Minsik and Tulsiani, Shubham},
    booktitle={CVPR},
    year={2026}
}
```


