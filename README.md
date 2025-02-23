# **Problem statement**
-------------------------------
**Problem Statement**  

Africa’s rapid population growth has worsened employment challenges, leading to high
unemployment, underemployment, and workforce instability. Reports from the ILO and AfDB 
highlight issues like skill mismatches, poor retention, and brain drain, yet existing research often 
addresses these problems in isolation without predictive solutions. Traditional workforce management remains 
reactive, failing to anticipate retention risks. This project aims to improve workforce stability by enhancing machine learning 
applications in retention modeling, providing data-driven insights to reduce turnover and inform better employment policies.

## Data source: **Kaggle**
------------------------------

My dataset was found: [HR analytics](https://www.kaggle.com/datasets/rishikeshkonapure/hr-analytics-prediction)

# Instance summary
|Instance  | Regularizer |   Optimizer |  Early Stopping |Number of layer| Accuracy | F1 score | Recall | Precision |                                             
|----------|-------------|-------------|-----------------|---------------|----------|----------|--------|-----------|
|Default   |     -       |     RMSprop    |       NO        |       3       |  0.7547  | 0.5455   | 0.5417 | 0.5493    |
|Model_1   |    l1       |    RMSpop   |       Yes       |       3       | 0.7698   | 0.5344   | 0.4861 | 0.5932    |
|Model_2   |     l2      |    Adam     |      Yes        |       3       | 0.7660   | 0.5000   | 0.4306 | 0.5962    |
|Model_3   |   l1_l2     |    RMSpop   |      Yes        |       3       | 0.7698   | 0.5344   | 0.4861 | 0.5932    |
|Model_4   | **penalty** | **tol**     | **n_jobs**      | **solver**    |  -       |  -       |   -    |    -      |
|logistic                                                                                                           |
|regression|     l1      |     0.001   |       -1        | liblinear     |  0.7962  | 0.5846   | 0.5278 | 0.6552    |

# Discussion
----------------------------------------
During my training process, I evaluated both neural network models and traditional machine learning methods to compare their effectiveness in classification tasks.

Neural Network Models:
I developed four neural network models with the same architecture (3 layers) but varied regularization techniques, optimizers, and early stopping strategies to observe their impact on performance.

The default model (Adam, no regularization, no early stopping) performed well, achieving an accuracy of 0.7547 and an F1-score of 0.5455.
L1 regularization with RMSpop (Model_1) and L1_L2 regularization with RMSpop (Model_3) both improved accuracy slightly (0.7698) but had a lower F1-score (0.5344).
L2 regularization with Adam (Model_2) resulted in slightly lower accuracy (0.7660) and an even lower F1-score (0.5000), indicating a trade-off in regularization effectiveness.
Traditional Machine Learning Model:
I also experimented with Logistic Regression, specifically using L1 regularization with the liblinear solver, achieving the highest accuracy of 0.7962 and the best F1-score of 0.5846 among all models.

Key Takeaways:
The best neural network model was the first model (Adam optimizer, no regularization, no early stopping) as it had the best balance between accuracy and loss reduction.
Adding regularization (L1, L2, L1_L2) had mixe~d effects—while accuracy slightly improved in some models, F1-scores and recall suffered.
Logistic Regression outperformed all neural networks in accuracy and F1-score, showing that traditional ML methods can still be highly effective depending on the problem.
Further fine-tuning hyperparameters (e.g., learning rate, batch size) might improve the neural network models, particularly for models that underperformed due to regularization constraints.

# here is the video explaining the instance table : [The Video](https://drive.google.com/file/d/1XSGOHDXjuYIYB7-LGetFt5LANYv26ITk/view?usp=drive_link)
