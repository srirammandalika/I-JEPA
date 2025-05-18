# I-JEPA

A simple Image-Joint Embedding Predictive Architecture (I-JEPA) - Modified implementation on CIFAR - 10 dataset.


### Simple Testing

```
python -m src.inference.visualize_masks
```

### Simple Training
Please not that you have to change the 'train_root' and 'val_root' directories. You can also adjust the batch size, epochs and max_per_class (max number samples used per class) as per your GPU setup. The code supports with Apple M1/2/3/4 - MPS and CUDA toolkit.

```
python -m src.training.train_student \
  --train_root "/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/train/" \
  --val_root   "/Users/srirammandalika/Downloads/Minor/CIFAR-10 data/cifar10/test/" \
  --batch_size 64 \
  --epochs     30 \
  --max_per_class 1000 \
  --lambda_pixel 0.1
```
