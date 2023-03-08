## Assignmnet-1: Image Classification in Cifar-10 with k-Nearest Neighbour
[Open in Kaggle](https://www.kaggle.com/code/alphapii/knn-cvpr)

![KNN gif](https://raw.githubusercontent.com/sadhiin/cvpr-spring-2023/main/others/images/knn.gif)

Cifar10 is a well-known dataset composed of 60,000 colored 32x32 images: 50,000 as a training set and 10,000 as a test set.
Nearest Neighbor is a machine learning algorithm that compares the training data based on their similarities to a training set accompanied with labels. This algorithm is mostly referred to as "k-nearest neighbor, where "k" refers to the number of neighbors that will be compared to predict the unknown data.

Image classification using KNN is a method of classifying images in the Cifar10 dataset using the K-Nearest Neighbor algorithm. Where the distance between all the training images and the predicted label is calculated by the most closely related training image label,

The complete process of the image classification is implemented by considering both distance matrices L1 and L2 when calculating the distances of images. The total of 50,000 images are scattered within the 5 folds of the list for cross-validation and choosing the best value of k (the number of neighbors to consider).

After implementing the KNN image classifier for the Cifar10 image set, the maximum accuracy using L1 (Manhattan distance) was 22.9%, taking the "8" neighbor voting, and using L2 (Euclidian distance) was 22.9% for 100 values of k.

Conceptually, the implementation of this classifier was simple and straightforward. Which was it that we have to compare the unknown label image with all known images, among them the most accurate label image? But there are some drawbacks to the algorithm. Those are

Repetitive task: The test image will be compared with all the test images, and distance will be calculated. This process took far too long for the algorithm to be developed. This means that in real-life implementation, getting the result or prediction will take too much time. This is not reliable to deploy and use in production. We always want the minimal time to get the predicted result.

Low accuracy: In the algorithm, pixel by pixel is calculated. In a scenario where the same object has a different background or view point, the algorithm might miss the object and consider the background due to the high number of pixels, leading to a missed classification of the data.

Overall, the implementation is a good improvement in image classification. Also, this algorithm shows the importance of deep learning for classifying the images. Where the training time of the model might take a long time (depending on the problem and datasets), the actual testing time is quick to get the prediction.