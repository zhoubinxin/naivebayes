`alpha` 是朴素贝叶斯分类器中的一个超参数，用于拉普拉斯平滑（Laplace smoothing）或加法平滑（Additive smoothing）。在处理文本分类任务时，拉普拉斯平滑帮助解决了零概率问题。具体来说，零概率问题是指如果某个特征在训练数据中没有出现过，那么在测试数据中包含这个特征的样本将无法被正确分类，因为概率计算中会出现零的情况。

### 拉普拉斯平滑

在朴素贝叶斯算法中，条件概率的计算公式为：

\[ P(x_i | y) = \frac{\text{count}(x_i \cap y) + \alpha}{\text{count}(y) + \alpha \cdot |V|} \]

其中：
- \( P(x_i | y) \) 是在给定类别 \( y \) 的条件下，特征 \( x_i \) 的概率。
- \( \text{count}(x_i \cap y) \) 是类别 \( y \) 中特征 \( x_i \) 出现的次数。
- \( \text{count}(y) \) 是类别 \( y \) 出现的次数。
- \( |V| \) 是特征的总数（词汇表大小）。
- \( \alpha \) 是平滑参数。

### `alpha` 的作用

1. **避免零概率**：如果某个特征在训练集中从未出现过，其条件概率会变成零。通过添加 `alpha`，即使特征未出现，其概率也不会是零。
2. **调节平滑强度**：`alpha` 的值越大，平滑效果越强。通常情况下，`alpha` 取值为 1 被称为拉普拉斯平滑。值为 0.1 或 0.5 则代表不同强度的平滑。

### 在代码中的应用

在我们自定义的 `SimpleNaiveBayes` 类中，`alpha` 作为一个参数传递给模型，并在计算条件概率时使用。

### 示例代码

```python
class SimpleNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            self.feature_log_prob_[idx] = np.log((X_c.sum(axis=0) + self.alpha) / (X_c.sum() + self.alpha * n_features))
    
    def predict(self, X):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def get_params(self, deep=True):
        return {"alpha": self.alpha}
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
```

在超参数搜索中，我们尝试不同的 `alpha` 值，以找到对数据集最优的参数：

```python
param_grid = {'alpha': [0.5, 1.0]}
grid_search = SimpleGridSearchCV(SimpleNaiveBayes(), param_grid, cv=3)
grid_search.fit(X_train_vec, y_train)

best_params = grid_search.best_params_
print(f"最佳参数: {best_params}")

best_model = SimpleNaiveBayes(**best_params)
best_model.fit(X_train_vec, y_train)
```

通过调整 `alpha` 值，可以观察到模型在不同平滑参数下的性能表现，从而选择最适合当前数据集的平滑参数。