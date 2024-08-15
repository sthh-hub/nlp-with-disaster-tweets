# cnn-cancer-detection


## 1. Project Overview

---------------------
The goal of the Histopathologic Cancer Detection project is to classify tissue sample images to determine whether they contain cancerous cells. This problem is framed as a binary classification task, where each image is assigned one of two labels:

• Label 0: Represents non-cancerous (benign) tissue

• Label 1: Represents cancerous (malignant) tissue



**Dataset Breakdown:**

The dataset used for this project is divided into two main parts:

1.	Training Dataset:

• Comprises a total of 220,025 images.

• Label Distribution:

• Non-cancerous (Label 0): 130,908 images

• Cancerous (Label 1): 89,117 images

• This dataset is used to train the model, allowing it to learn and distinguish between cancerous and non-cancerous tissues.

2.	Test Dataset:

• Contains 57,458 images that are unlabeled.

• The model’s performance will be evaluated based on its ability to correctly classify these images.

---------------------

## 2. EDA

1) CNN Model Overview


	**Architecture**

	• Convolutional Layers: The model consists of three convolutional layers, each followed by a ReLU activation function and max-pooling for downsampling.

	• Conv1: 32 filters, each of size 3x3, to capture low-level features like edges and textures.

	• Conv2: 64 filters of size 3x3, designed to extract more complex patterns from the feature maps produced by the first layer.

	• Conv3: 128 filters of size 3x3, focusing on even more abstract features as the depth of the network increases.

	• Fully Connected Layers: The network includes two fully connected layers that integrate the extracted features and produce the final output.

	• FC1: 512 units, serving as a dense layer that synthesizes the learned features into higher-level representations.

	• FC2: 1 unit, acting as the output layer to produce a single probability score for binary classification.

	• Dropout: A dropout layer with a rate of 0.25 is applied after the first fully connected layer to mitigate the risk of overfitting by randomly disabling 25% of the neurons during training.

	• Activation Function: A sigmoid activation function is used in the output layer to map the final score to a probability value between 0 and 1, suitable for binary classification.
	
	
	**Reasoning**

	• Feature Extraction: The convolutional layers are responsible for extracting hierarchical features from the input images, progressively capturing more complex patterns as the layers deepen.

	• Non-Linearity: ReLU (Rectified Linear Unit) activation functions introduce non-linearity into the model, enabling it to learn complex patterns that are not linearly separable.

	• Downsampling: Max-pooling layers are used to reduce the spatial dimensions of the feature maps, which decreases computational requirements and helps the model become invariant to small translations in the input images.

	• Regularization: The dropout layer acts as a form of regularization, reducing the likelihood of the model overfitting to the training data by preventing co-adaptation of neurons.
	
	• Classification: The fully connected layers synthesize the information extracted by the convolutional layers, and the final sigmoid activation function outputs a probability, allowing the model to make a binary decision between cancerous and non-cancerous tissue.


2) DenseNet Model (Transfer Learning)


	**Architecture**
	•	Pre-trained Backbone: DenseNet121 with pre-trained weights from ImageNet.
	•	Modification: Replace the final classifier layer to adapt to the binary classification task by using a single fully connected layer followed by a sigmoid activation function.


	**Reasoning**
	•	Pre-trained Features: Utilize features learned from the extensive ImageNet dataset, allowing the model to leverage complex and generalizable feature representations. This often results in faster convergence and improved performance, especially when training on smaller datasets.
	•	Dense Connections: The dense connections between layers ensure efficient gradient flow and feature reuse, which leads to a more compact and efficient model. This architecture helps in learning detailed patterns and can achieve high accuracy with fewer parameters compared to traditional architectures like ResNet.
---------------------

## 4. Results and Analysis

To compare the performance of the CNN and DenseNet models for Histopathologic Cancer Detection, the following steps were taken:

1. Hyperparameter Tuning

    We defined a range of hyperparameters, including learning rate, batch size, and dropout rate, for both models. Grid search or random search was used to identify the optimal combination of these hyperparameters.


2. Model Training

    Both models were trained using the selected hyperparameters. Metrics such as accuracy, F1 score, and AUC-ROC were monitored to evaluate model performance.


3. Evaluation and Comparison

    The trained models were evaluated on the validation set, and their performances were compared based on key metrics:
    • Accuracy: Overall classification correctness.
    • F1 Score: Balances precision and recall.
    • AUC-ROC: Assesses the model’s ability to distinguish between classes.


4. Discussion

    The results showed differences in performance between the CNN and DenseNet models. The reasons for these differences were analyzed, and potential improvements, such as further hyperparameter tuning or enhanced data augmentation, were suggested.


**CNN Results**

Experiments were conducted with various learning rates and batch sizes for the CNN model. The key observations are as follows:
• Learning Rate 0.001, Batch Size 32: The model shows good convergence in both training and validation losses, indicating strong performance.

• Learning Rate 0.001, Batch Size 64: Performance is similar, but the convergence is slightly slower compared to batch size 32.

• Learning Rate 0.0001, Batch Size 32: The model converges more slowly, suggesting that the learning rate might be too low.

• Learning Rate 0.0001, Batch Size 64: Similar to the previous setting, with even slower convergence.

Overall, the combination of a learning rate of 0.001 and a batch size of 32 proved to be the most effective for the CNN model.


**DenseNet Results**

We performed similar experiments with the DenseNet model. The observations include:
• Learning Rate 0.001, Batch Size 32: The model converges well but exhibits slight fluctuations in the validation loss.

• Learning Rate 0.001, Batch Size 64: The model maintains good performance, similar to the smaller batch size, with minor oscillations.

• Learning Rate 0.0001, Batch Size 32: The model shows stable but slower convergence, indicating a learning rate that might be too low.

• Learning Rate 0.0001, Batch Size 64: Similar results to the previous combination, with slower overall convergence.

For the DenseNet model, a learning rate of 0.001 with a batch size of 32 also appears to be the optimal configuration.

----------------

## 5. Conclusion 

### Learnings and Takeaways

From these experiments, I learned that:

1. **Learning Rate and Batch Size**: A learning rate of 0.001 combined with a batch size of 64 provided the most consistent and effective performance for the CNN model. This configuration allowed the model to converge efficiently while maintaining stability in both training and validation losses.
2. **Model Stability**: The DenseNet model exhibited instability during training, with fluctuating losses that suggest it may require further tuning or alternative regularization strategies. The CNN model, on the other hand, demonstrated greater reliability across different configurations.
3. **Training Time vs. Performance**: The best-performing CNN configuration showed that a slightly larger batch size (64) could still achieve optimal results without a significant increase in training time, balancing both efficiency and performance.

### What Helped Improve Performance

1. **Optimal Hyperparameters**: Identifying the right combination of learning rate and batch size was key to enhancing the performance of the CNN model, leading to more stable and lower loss values.
2. **Simplicity of Architecture**: The straightforward architecture of the CNN, combined with careful tuning of hyperparameters, proved more effective and stable compared to the more complex DenseNet architecture in this particular task.

### What Did Not Help

1. **DenseNet's Instability**: Despite its sophisticated design, DenseNet struggled with stability, which hindered its overall effectiveness in this experiment. This indicates that more advanced architectures are not always the best choice without careful tuning.
2. **Higher Learning Rates**: While experimenting with a learning rate of 0.01, the model failed to achieve better performance, leading to higher losses and less stable training.

### Future Improvements

1. **Focus on Regularization**: Implementing additional regularization techniques, such as more aggressive dropout or weight decay, could help improve the stability of models like DenseNet and reduce overfitting in CNN.
2. **Learning Rate Scheduling**: Introducing learning rate schedulers, such as ReduceLROnPlateau, might optimize the learning process and prevent sudden fluctuations in loss.
3. **Explore Different Architectures**: While CNN performed well, exploring other architectures like EfficientNet or further tuning DenseNet could yield even better results.
4. **Advanced Hyperparameter Tuning**: Utilizing more sophisticated hyperparameter optimization methods, such as Bayesian optimization, could further refine the model's performance and lead to more efficient training.

By focusing on these areas for improvement, future experiments can aim for even greater performance and generalization capabilities in image classification tasks like histopathologic cancer detection.

