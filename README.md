# EffShuff_Block(Efficient ShuffleNet Block)

## Summary
> The EffShuff-Block model is designed to be lightweight while maintaining high classification performance.The input feature map is divided into two parts and processed differently, with one half undergoing a lightweight convolution, and the other half undergoing average pooling. The EffShuff-Transition component performs pooling by shuffling after performing lightweight convolution, resulting in a 57.9% reduction in computational cost compared to ShuffleNetv2, a well-known optimization for lightweight CNN models. In the experiments, the proposed EffShuff-Block model achieved 96.975% accuracy in age and gender classification, which is 5.83% higher than the state-of-the-art. The EffShuff-Dense-Block (Efficient shuffle dense block) model, which incorporates Dense Block to further emphasize low-level features, achieved 97.63% accuracy. Additionally, the results of the fine-grained image classification experiment demonstrate that the proposed EffShuff-Block and EffShuff-Dense-Block models have better classification performance with a smaller model size.
![그림5](https://user-images.githubusercontent.com/20642014/217741663-d2e6e01a-f0df-4d7f-b141-e824c163a8ae.png)
![그림6_6](https://user-images.githubusercontent.com/20642014/217741688-6f9965ca-65ee-4707-868f-2ab0943a7955.png)

## Dataset
1. The Audience dataset is available at https://talhassner.github.io/home/projects/Adience/Adience-data.html.
2. The Butterfly & Moths dataset is available https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species.
3. the NABirds dataset is available at https://dl.allaboutbirds.org/nabirds.
![그림8](https://user-images.githubusercontent.com/20642014/217741705-79c1a640-ccb7-48e2-aeb9-df4dde058df6.png)
![그림9](https://user-images.githubusercontent.com/20642014/217741711-5fb35dda-09e5-44d6-bdce-a7f048cf9254.png)
![그림10](https://user-images.githubusercontent.com/20642014/217741716-61a2bea7-66dd-4e6b-89d7-b0a4c18b6c4b.png)
