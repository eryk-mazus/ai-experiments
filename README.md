# ai-experiments
implementing generative stuff from scratch

## some samples😇:
* Sampling from VAE model trained on MNIST:
![vae_generation](https://github.com/eryk-mazus/ai-experiments/assets/21311210/e6bf4414-c0d5-460c-8917-2de14ddcf035)

* Sampling integers from Conditional VAE trained on MNIST:
![conditional_vae_generation](https://github.com/eryk-mazus/ai-experiments/assets/21311210/8c0382e7-c313-43b1-8e5d-275b5f840038)

* celeba reconstructions using [VQ-VAE](https://github.com/eryk-mazus/ai-experiments/blob/main/ai_experiments/vq_vae/vq_vae.py) (examples from validation dataset):
![Figure_1rec](https://github.com/eryk-mazus/ai-experiments/assets/21311210/609859a3-4b38-4bde-8c68-cf144851f681)

## setup

```shell
git clone https://github.com/eryk-mazus/ai-experiments.git
cd ai-experiments
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
echo "Ready to go"
```
