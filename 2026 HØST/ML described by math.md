## Strip away the buzzwords first

Linear regression is this and only this: you're finding a function

```
ŷ = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

- `x₁...xₙ` — your **features** (network activity metrics: packets/sec, bytes sent, connections open, etc.)
- `w₁...wₙ` — the **weights**: how much each feature matters. This is what you're solving for.
- `b` — the **bias** (intercept): the baseline output when all features are zero.
- `ŷ` (y-hat) — your **prediction** (estimated power consumption)
- `y` — the **actual** value from your dataset (ground truth)

That's the entire model. "Training" means: find the `w`s and `b` that make `ŷ` close to `y` across all your data rows.

## How "close" is measured — the cost function

You need a single number that tells you how wrong the model currently is. The standard one for linear regression is **Mean Squared Error (MSE)**:

```
J(w, b) = (1/m) * Σ(ŷᵢ - yᵢ)²
```

- `m` = number of training examples (rows in your dataset)
- You square the error so positive and negative errors don't cancel out, and so bigger errors are punished more
- `J` is just the conventional letter for "cost function" — no deeper meaning

This `J` is a function of your weights and bias. Training = finding the `w, b` that **minimize** `J`.

## How you minimize it — gradient descent

You can't just guess. You compute the **gradient** (the vector of partial derivatives of `J` with respect to each weight and the bias) — this tells you the direction of steepest *increase* in error. You step in the opposite direction.

The derivatives (this is the calculus you already have):

```
∂J/∂wⱼ = (2/m) * Σ(ŷᵢ - yᵢ) * xᵢⱼ
∂J/∂b  = (2/m) * Σ(ŷᵢ - yᵢ)
```

Update rule, applied every iteration:

```
wⱼ := wⱼ - α * ∂J/∂wⱼ
b  := b  - α * ∂J/∂b
```

- `α` (alpha) = **learning rate** — how big a step you take. Too big → diverges/oscillates. Too small → painfully slow.
- One full pass of computing gradients over all `m` examples and updating = one **epoch** (or "iteration," people use these loosely)
- **Convergence** = when `J` stops decreasing meaningfully between epochs

That's it. No magic. Everything else in ML (neural nets, etc.) is this same loop with a fancier `ŷ`.

## One more thing you'll need: feature scaling

Your features will be wildly different scales — "bytes sent" might be in the millions, "open connections" might be 0–50. Gradient descent converges badly (or not at all) with unscaled features because the cost surface becomes a narrow ravine. So you **normalize**:

```
x_scaled = (x - mean) / std_dev      // standardization
```

You do this per-feature, store the mean/std, and remember to apply the same transform to any new data you predict on later (and un-scale your final predictions/weights if you want them in original units).

## Now, the C++ — no linear algebra library, just `std::vector`

Here's a full from-scratch implementation. I'm keeping the math explicit (loops, not matrix notation) so every operation maps back to the formulas above.

```cpp
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;

struct Dataset {
    vector<vector<double>> X; // rows of features
    vector<double> y;         // targets
    int numFeatures;
};

// --- Load CSV: assumes last column is target, rest are features, no header handling shown for brevity ---
Dataset loadCSV(const string& path) {
    Dataset d;
    ifstream file(path);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        d.y.push_back(row.back());
        row.pop_back();
        d.X.push_back(row);
    }
    d.numFeatures = d.X[0].size();
    return d;
}

// --- Feature scaling (standardization) ---
void standardize(vector<vector<double>>& X, vector<double>& means, vector<double>& stds) {
    int m = X.size();
    int n = X[0].size();
    means.assign(n, 0.0);
    stds.assign(n, 0.0);

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) means[j] += X[i][j];
        means[j] /= m;
    }
    for (int j = 0; j < n; j++) {
        double sumSq = 0.0;
        for (int i = 0; i < m; i++) sumSq += (X[i][j] - means[j]) * (X[i][j] - means[j]);
        stds[j] = sqrt(sumSq / m);
        if (stds[j] == 0) stds[j] = 1.0; // avoid divide-by-zero on constant columns
    }
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            X[i][j] = (X[i][j] - means[j]) / stds[j];
}

// --- Prediction for a single row: ŷ = w·x + b ---
double predict(const vector<double>& x, const vector<double>& w, double b) {
    double result = b;
    for (size_t j = 0; j < x.size(); j++) result += w[j] * x[j];
    return result;
}

// --- Cost function: MSE ---
double computeCost(const vector<vector<double>>& X, const vector<double>& y,
                    const vector<double>& w, double b) {
    int m = X.size();
    double totalError = 0.0;
    for (int i = 0; i < m; i++) {
        double error = predict(X[i], w, b) - y[i];
        totalError += error * error;
    }
    return totalError / m;
}

// --- Gradient descent ---
void trainGradientDescent(const vector<vector<double>>& X, const vector<double>& y,
                           vector<double>& w, double& b,
                           double alpha, int epochs) {
    int m = X.size();
    int n = X[0].size();

    for (int epoch = 0; epoch < epochs; epoch++) {
        vector<double> dw(n, 0.0);
        double db = 0.0;

        // Accumulate gradients over all examples
        for (int i = 0; i < m; i++) {
            double error = predict(X[i], w, b) - y[i]; // (ŷ - y)
            for (int j = 0; j < n; j++) {
                dw[j] += error * X[i][j];
            }
            db += error;
        }

        // Average and apply update
        for (int j = 0; j < n; j++) {
            dw[j] = (2.0 / m) * dw[j];
            w[j] -= alpha * dw[j];
        }
        db = (2.0 / m) * db;
        b -= alpha * db;

        if (epoch % 100 == 0) {
            cout << "Epoch " << epoch << " | Cost: " << computeCost(X, y, w, b) << endl;
        }
    }
}

int main() {
    Dataset data = loadCSV("network_power.csv");

    vector<double> means, stds;
    standardize(data.X, means, stds);

    vector<double> w(data.numFeatures, 0.0); // init weights to 0
    double b = 0.0;

    double alpha = 0.01;
    int epochs = 1000;

    trainGradientDescent(data.X, data.y, w, b, alpha, epochs);

    cout << "\nFinal weights: ";
    for (double wj : w) cout << wj << " ";
    cout << "\nFinal bias: " << b << endl;

    return 0;
}
```

## Why it's built this way (the parts people gloss over)

- **Weights init to 0**: fine for plain linear regression (the cost surface is convex — one global minimum, no symmetry-breaking issue like in neural nets).
- **Batch gradient descent**: this version sums over *all* `m` rows before updating once per epoch. That's "batch." If you update after every single row, that's **stochastic gradient descent (SGD)**; a chunk at a time is **mini-batch**. Same math, different granularity.
- **Why `2/m` and not `1/m`**: comes directly from differentiating the squared error — the 2 falls out of the power rule. Some implementations fold it into `α` instead; doesn't matter which, as long as you're consistent.
- **The cost printout every 100 epochs**: this is how you'll *see* convergence — if cost stops dropping, you either converged or your learning rate is wrong.

## What to check once it runs

- Plot cost vs. epoch (even just print to a file, chart it) — it should be a smooth decreasing curve. If it oscillates or explodes, lower `α`. If it decreases extremely slowly, raise it.
- Evaluate on held-out data you didn't train on, using **RMSE** (root of MSE, in your original units) or **R²** (fraction of variance explained) — these are the standard scoring terms you'll see everywhere.

Want me to extend this next with train/test splitting and an R² calculation, or first walk through how the normal equation (closed-form solution, no gradient descent needed) would solve this same problem so you can compare the two approaches?