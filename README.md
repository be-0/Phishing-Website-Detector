## 📌 Overview

This project implements a phishing website detection system using a feed-forward neural network trained on structured website features from the UCI Phishing Websites dataset. The dataset contains 30 pre-engineered, rule-based features describing URL and website characteristics, with labels indicating phishing, suspicious, or legitimate websites.

The model is trained using mini-batch gradient descent and optimized with cross-entropy loss. Multiple optimizers (SGD and Adam) and regularization techniques such as weight decay, dropout, and batch normalization were explored to improve generalization. A majority vote classifier and simpler neural network variants were implemented as baselines for comparison.

Performance was evaluated using accuracy, precision, recall, and F1-score on a validation set. The final model achieved approximately 95% validation accuracy, demonstrating strong generalization with minimal overfitting. Ablation studies showed that proper mini-batch training, increased hidden-layer capacity, and adaptive optimization significantly improved learning stability and performance.

Overall, the project demonstrates that a relatively simple neural network can effectively classify phishing websites when trained on well-designed tabular features, while also highlighting the impact of architectural and optimization choices on model performance.

---
