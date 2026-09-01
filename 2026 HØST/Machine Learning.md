# Machine Learning
## The core idea
Machine learning is building systems that improve at a task by learning patterns from data, rather than being explicitly programmed with rules. Instead of writing "if X then Y," you show the system many examples and it figures out the mapping itself.

## The three main paradigms
1. **Supervised learning** - You have labeled data (input -> correct output) and train a model to predict the output for new inputs. Examples: spam detection, house price prediction, image classification.
2. **Unsupervise learning** - No labels; the model finds structure on its own. Examples: clustering customers by behavior, dimensionality reduction, anomaly detection.
3. **Reinforcement learning** - An agent learns by taking actions in an environment and getting rewards or penalties. Used in game-playing AI, robotics, and increasingly in fine-tuning language models.

## Key building blocks
- **Features** - the input variables the model uses
- **Model** - the mathematical function that maps inputs to outputs (linear regression, decision tree, neural network, etc.)
- **Loss function** - measures how wrong the model's predictions are
- **Training** - adjusting the model's parameters (usually via gradient descent) to minimize the loss
- **Overfitting/ underfitting** - the central tension: a model that memorizes training data does poorly on new data (overfit); one that's too simple misses real patterns (underfit)
- **Train/ validation/ test split** - how you check a model generalizes rather than just memorizes

## Commonalgorithm families (roughly simple -> complex)
1. Linear/logistic regression
2. Decision trees and random forests
3. Support vector machines
4. Gradient boosting (XGBoost, LightGBM) - dominant for tabular data
5. Neural network - dominant for images, text, audio, and complex pattern recognition
6. Transformers - the architecture behind modern LLMs, built on "attention" mechanisms

## Deep learning specifically
A subset of ML using multi-layered neural networks. It's what powers image recognition, speech recognition, and language models like the one you're talking to. Key concepts: layers, weights, backpropagation, activation functions, and - for modern large models - pretraining on huge datasets followed by fine-tuning.