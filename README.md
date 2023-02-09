# EffShuff_Block(Efficient ShuffleNet Block)

## Summary
> The EffShuff-Block model is designed to be lightweight while maintaining high classification performance.The input feature map is divided into two parts and processed differently, with one half undergoing a lightweight convolution, and the other half undergoing average pooling. The EffShuff-Transition component performs pooling by shuffling after performing lightweight convolution, resulting in a 57.9% reduction in computational cost compared to ShuffleNetv2, a well-known optimization for lightweight CNN models. In the experiments, the proposed EffShuff-Block model achieved 96.975% accuracy in age and gender classification, which is 5.83% higher than the state-of-the-art. The EffShuff-Dense-Block (Efficient shuffle dense block) model, which incorporates Dense Block to further emphasize low-level features, achieved 97.63% accuracy. Additionally, the results of the fine-grained image classification experiment demonstrate that the proposed EffShuff-Block and EffShuff-Dense-Block models have better classification performance with a smaller model size.
![그림5](https://user-images.githubusercontent.com/20642014/217741663-d2e6e01a-f0df-4d7f-b141-e824c163a8ae.png)
![그림6_6](https://user-images.githubusercontent.com/20642014/217741688-6f9965ca-65ee-4707-868f-2ab0943a7955.png)

## Dataset
1. The Audience dataset is available at https://talhassner.github.io/home/projects/Adience/Adience-data.html.
2. The Butterfly & Moths dataset is available https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species.
3. the NABirds dataset is available at https://dl.allaboutbirds.org/nabirds.

## Training
| Task | Training Set | Val_acc | Params | Flops |
|---|:---:|:---:|:---:|:---:|
| `classification` | The Audience dataset(Age) | 96.37 | 1.13M | 4.85G |
| `classification` | The Audience dataset(Gender) | 97.58 | 1.13M | 4.85G |
| `Fine-grained classification` | Butterfly & Moths dataset | 97.70 | 1.21M | 4.85G |
| `Fine-grained classification` | NABirds dataset | 88.26 | 1.64M | 9.70G |

## Result
- Age

![AGE_accuracy](https://user-images.githubusercontent.com/20642014/217746693-c29b06ba-22a5-4410-80c7-ac5ca0a84b3f.png)
![AGE_loss](https://user-images.githubusercontent.com/20642014/217746701-a69757fb-734e-4010-b8fb-f092ce9c5772.png)

- Gender
![Gender_acc](https://user-images.githubusercontent.com/20642014/217746712-a4e26940-e7d7-4e64-98ff-00ab3d809194.png)
![Gender_loss](https://user-images.githubusercontent.com/20642014/217746720-bbb039e2-5c8b-4daa-833f-8561e9d5eb8c.png)

- Butterfly and moths
![butterfly acc](https://user-images.githubusercontent.com/20642014/217746726-eba77d9f-b2ab-4adc-af92-8d9738fdfda9.png)
![butterfly loss](https://user-images.githubusercontent.com/20642014/217746732-8f30b507-4fa4-47c5-a115-72d517cbd5f1.png)

- NA Birds
![NA birds acc](https://user-images.githubusercontent.com/20642014/217748693-ce2b205e-f09c-4e06-bfc4-d831a49541c2.png)
![NA birds loss](https://user-images.githubusercontent.com/20642014/217748705-927d3335-0a78-4c41-8df4-4f74371c790c.png)
