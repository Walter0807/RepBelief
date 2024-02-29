# Language Models Represent Beliefs of Self and Others

<p align="center">
<a href="https://arxiv.org/pdf/2402.18496.pdf", target="_blank">
<img src="https://walter0807.github.io/RepBelief/assets/buttons_paper.png"alt="arXiv" style="width: 120px;"></a>
<a href="https://walter0807.github.io/RepBelief", target="_blank">
<img src="https://walter0807.github.io/RepBelief/assets/buttons_cursor.png" alt="Project Page" style="width: 120px;"></a>
</p>


<div style="margin: 30px auto; display: block; text-align: center;">
    <p align="center">
      <img src="https://walter0807.github.io/RepBelief/assets/teaser.jpg" 
           style="width: 50%; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); display: inline-block;">
    </p>
</div>


This repository provides the code for the paper "Language Models Represent Beliefs of Self and Others". It shows that LLMs internally represent beliefs of themselves and other agents, and manipulating these representations can significantly impact their Theory of Mind reasoning capabilities.


## Installation

```
conda create -n lm python=3.8 anaconda
conda activate lm
# Please install PyTorch according to your CUDA version.
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirements.txt
```

Then download the language models (e.g. [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [deepseek-llm-7b-chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat)) to `models/`. You can also specify the file paths in `lm_paths.json`.



## Extract Representations

```bash
sh scripts/save_reps.sh 0_forward belief
sh scripts/save_reps.sh 0_forward action
sh scripts/save_reps.sh 0_backward belief
```



## Probing

Binary:

```bash
python probe.py --belief=protagonist --dynamic=0_forward --variable belief 
python probe.py --belief=oracle --dynamic=0_forward --variable belief

python probe.py --belief=protagonist --dynamic=0_forward --variable action 
python probe.py --belief=oracle --dynamic=0_forward --variable action

python probe.py --belief=protagonist --dynamic=0_backward --variable belief 
python probe.py --belief=oracle --dynamic=0_backward --variable belief
```



Multinomial:

```bash
python probe_multinomial.py --dynamic=0_forward --variable belief
python probe_multinomial.py --dynamic=0_forward --variable action
python probe_multinomial.py --dynamic=0_backward --variable belief
```



## BigToM Evaluation

```bash
sh scripts/0_forward_belief.sh
sh scripts/0_forward_action.sh
sh scripts/0_backward_belief.sh
```



## Intervention

Intervention for the *Forward Belief* task:

```bash
sh scripts/0_forward_belief_interv_oracle.sh
sh scripts/0_forward_belief_interv_protagonist.sh
sh scripts/0_forward_belief_interv_o0p1.sh
```



Cross-task intervention:

```bash
sh scripts/cross_0_forward_belief_to_forward_action_interv_o0p1.sh
sh scripts/cross_0_forward_belief_to_backward_belief_interv_o0p1.sh
```



## Citation

```bibtex
@article{zhu2024language,
  title={Language Models Represent Beliefs of Self and Others},
  author={Zhu, Wentao and Zhang, Zhining and Wang, Yizhou},
  year={2024}
}
```
