2022年度映像メディア学課題用レポジトリです。

HMS-Netを実装しました。

[Z. Huang, J. Fan, S. Cheng, S. Yi, X. Wang and H. Li, "HMS-Net: Hierarchical Multi-Scale Sparsity-Invariant Network for Sparse Depth Completion," in IEEE Transactions on Image Processing, vol. 29, pp. 3429-3441, 2020, doi: 10.1109/TIP.2019.2960589.](https://arxiv.org/abs/1808.08685)

ブログ：https://ko-pen.hatenablog.com/

train

```
python train.py --epoch 50 --dataset PATH_TO_DATASET --gpu 0
```

test

```
python test.py --dataset PATH_TO_DATASET --gpu 0 --model PATH_TO_MODEL
```
