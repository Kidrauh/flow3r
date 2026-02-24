<div align="center">
	<h1>Flow3r: Factored Flow Prediction for Visual Geometry Learning</h1>
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
	<!-- <a href=""><img src="https://img.shields.io/badge/arXiv-2512.10950-b31b1b" alt="arXiv"></a> -->
	<a href="https://flow3r-project.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
	<!-- <a href=""><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> -->
</div>

<div align="center">
    <img src="assets/teaser_video.gif" alt="overview">
</div>
	
<!-- ![teaser](https://raw.githubusercontent.com/QitaoZhao/QitaoZhao.github.io/main/research/E-RayZer/images/erayzer_teaser.png) -->

## Overview
Flow3r is a scalable framework for visual geometry learning that leverages flow prediction to guide learning using unlabeled monocular videos. We evaluate flow3r across diverse 3D benchmarks and demonstrate competitive or state-of-the-art performance, even surpassing supervised models trained with more labeled data.

## Quick Start

### 1. Create the environment
```bash
conda create -n flow3r python=3.11
conda activate flow3r

pip install -r requirements.txt
```
### 2. Download and place checkpoint
- `checkpoints/flow3r.bin`: Flow3r trained on ~834k video sequences.

Please fetch the checkpoints manually from Google Drive and drop the file into `checkpoints/`.
<!-- Please fetch the checkpoints manually from [Google Drive](https://drive.google.com/placeholder-flow3r-checkpoints) and drop the file into `checkpoints/`. -->

### 3. Launch the Gradio app
```bash
python gradio_app.py 
```

<!-- ## Citation
If you use Flow3r in academic or industrial research, please cite:

```bibtex

``` -->

<!-- 
## License
- **Code**: MIT License (see `LICENSE`).
- **Model weights**: Adobe Research License (see `LICENSE-WEIGHTS`).  The model weights are **not** covered by the MIT License. -->

## Acknowledgements
- Our work builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:
    - [Pi3](https://github.com/yyfz/Pi3?tab=readme-ov-file)
    - [VGGT](https://github.com/facebookresearch/vggt)
- We also thank the members of the [Physical Perception Lab](https://shubhtuls.github.io/) at CMU for their valuable discussions.


