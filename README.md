# visdiff-impl

This is the reimplementation of the paper "Describing differences in image sets with natural language" by Dunlap et al. The original paper can be found at <https://arxiv.org/abs/2312.02974>.

## Dataset

Download VissDiffBench dataset from `https://drive.google.com/file/d/1vghFd0rB5UTBaeR5rdxhJe3s7OOdRtkY/edit` and extract it to root directory.

## How to run

Set OpenAI API key in `.env` file and run the following commands:

```bash
uv sync
uv run python src/pipeline.py --dataset easy
```
