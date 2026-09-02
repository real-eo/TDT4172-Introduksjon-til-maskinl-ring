## Part 1: Train/test split + R²

The idea: you never evaluate a model on the data it trained on, because it'll look better than it actually is (it may have partially memorized noise). So you hold out a chunk of rows the model never sees during training.

```cpp
#include <algorithm>
#include <random>

// --- Split into train/test sets ---
void trainTestSplit(const Dataset& data, double testRatio,
                     vector<vector<double>>& Xtrain, vector<double>& ytrain,
                     vector<vector<double>>& Xtest, vector<double>& ytest) {
    int m = data.X.size();
    vector<int> indices(m);
    for (int i = 0; i < m; i++) indices[i] = i;

    // Shuffle so train/test aren't biased by row order (e.g. time-ordered logs)
    unsigned seed = 42; // fixed seed = reproducible splits
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    int testCount = static_cast<int>(m * testRatio);

    for (int i = 0; i < m; i++) {
        int idx = indices[i];
        if (i < testCount) {
            Xtest.push_back(data.X[idx]);
            ytest.push_back(data.y[idx]);
        } else {
            Xtrain.push_back(data.X[idx]);
            ytrain.push_back(data.y[idx]);
        }
    }
}

// --- R²: fraction of variance in y explained by the model ---
// R² = 1 - (SS_res / SS_tot)
// SS_res = sum of squared errors of YOUR model
// SS_tot = sum of squared errors of a "dumb" model that just predicts the mean every time
double computeR2(const vector<vector<double>>& X, const vector<double>& y,
                  const vector<double>& w, double b) {
    int m = y.size();
    double meanY = 0.0;
    for (double yi : y) meanY += yi;
    meanY /= m;

    double ssRes = 0.0, ssTot = 0.0;
    for (int i = 0; i < m; i++) {
        double pred = predict(X[i], w, b);
        ssRes += (y[i] - pred) * (y[i] - pred);
        ssTot += (y[i] - meanY) * (y[i] - meanY);
    }
    return 1.0 - (ssRes / ssTot);
}

// --- RMSE: same units as your target (watts, presumably) ---
double computeRMSE(const vector<vector<double>>& X, const vector<double>& y,
                    const vector<double>& w, double b) {
    return sqrt(computeCost(X, y, w, b)); // cost() is already MSE
}
```

**Reading R²**: 1.0 = perfect predictions. 0.0 = your model is exactly as good as just guessing the average power consumption every time, regardless of network activity. Negative = your model is actively worse than that dumb baseline (a real risk with a bad learning rate or non-standardized test data).

Updated `main()`:

```cpp
int main() {
    Dataset data = loadCSV("network_power.csv");

    vector<double> means, stds;
    standardize(data.X, means, stds); // scale BEFORE splitting is fine here since
                                       // we're computing stats from the whole set —
                                       // for rigor later, compute means/stds from
                                       // train only, then apply to test.

    vector<vector<double>> Xtrain, Xtest;
    vector<double> ytrain, ytest;
    trainTestSplit(data, 0.2, Xtrain, ytrain, Xtest, ytest); // 80/20 split

    vector<double> w(data.numFeatures, 0.0);
    double b = 0.0;

    trainGradientDescent(Xtrain, ytrain, w, b, 0.01, 1000);

    cout << "\nTrain R²: " << computeR2(Xtrain, ytrain, w, b) << endl;
    cout << "Test R²:  " << computeR2(Xtest, ytest, w, b) << endl;
    cout << "Test RMSE: " << computeRMSE(Xtest, ytest, w, b) << " watts" << endl;

    return 0;
}
```

If train R² is much higher than test R² (say 0.95 vs 0.4), that's **overfitting** — the model fit noise specific to the training rows. With plain linear regression on a small feature set this is less likely than with complex models, but it's the exact same diagnostic you'll use everywhere later.

I flagged one subtlety in the comment: technically, scaling should be fit on the training set only, then applied to test using those same means/stds — because in a real deployment you wouldn't have test data in advance to compute stats from. For your first pass it won't matter much; worth fixing once this works.

## Part 2: The normal equation — closed-form solution

Gradient descent is *iterative* — it approaches the answer step by step. But linear regression has a special property: the cost function `J` is a convex bowl shape (paraboloid), which means you can solve for the exact minimum directly with calculus, no iteration needed.

Setting the gradient to zero and solving algebraically (in matrix form, where `X` includes a column of 1s for the bias):

```
w = (XᵀX)⁻¹ Xᵀy
```

- `Xᵀ` = transpose of your feature matrix
- `(XᵀX)⁻¹` = matrix inverse — this is the expensive part, O(n³) where n = number of features
- This gives you the exact `w` (including bias, if you added the 1s column) in one shot

**When to use which:**
| | Gradient Descent | Normal Equation |
|---|---|---|
| Speed with many features (n > ~10,000) | Fast | Slow (matrix inversion is O(n³)) |
| Speed with many rows (m huge) | Scales fine | Fine, m doesn't affect the inverse |
| Needs feature scaling? | Yes | No |
| Needs to choose α? | Yes | No |
| Works for non-linear models later (neural nets)? | Yes | No — this trick only works for linear regression |

For your dataset (network metrics → power, probably tens of features, not thousands), the normal equation is very feasible and a great cross-check: train both, confirm they converge to nearly the same `w` and `b`. That's a strong correctness test for your gradient descent implementation.

Since you don't have a matrix library, implementing this from scratch means writing matrix multiply, transpose, and Gauss-Jordan inversion by hand — meaningfully more code than gradient descent. Do you want that written out too, or do you want to first get the gradient descent version running on your actual dataset and come back to the normal equation as a verification step once you have real numbers to compare against?