Excellent. We've reached **Day 6**, which is another foundational day. This lesson introduces concepts that are used in almost every real-world machine learning project.

Unlike previous days, Day 6 contains **four tightly connected topics**:

1. Multiple Linear Regression
2. Feature Scaling
3. Standardization
4. Normalization

Trying to cover all of them in one or two parts would make the lesson rushed. To stay aligned with your course standards (textbook quality, intuition before mathematics, and complete coverage), we'll split it into **five parts**.

---

# Week 1 — Day 6 Roadmap

### Part 1 — Multiple Linear Regression (Intuition & Mathematics)

* Why one feature is often insufficient.
* Multiple features and real-world examples.
* Hypothesis function with multiple variables.
* Geometry: line → plane → hyperplane.
* Matrix intuition.
* Feynman explanation.

---

### Part 2 — Cost Function & Gradient Descent for Multiple Features

* Extending the cost function.
* Partial derivatives with many weights.
* Gradient updates for multiple parameters.
* Vector notation.
* Python implementation (without vectorization first).

---

### Part 3 — Feature Scaling (One of the Most Important Practical Topics)

* Why Gradient Descent becomes slow without scaling.
* Units problem.
* Visual intuition.
* Types of feature scaling.
* When scaling is required.
* When scaling is unnecessary.

---

### Part 4 — Standardization vs Normalization

* Min-Max Normalization.
* Z-score Standardization.
* Mean and Standard Deviation.
* Choosing the right technique.
* Python implementation using NumPy.
* Real-world examples.

---

### Part 5 — Complete Implementation & Wrap-up

* Full Multiple Linear Regression implementation.
* Vectorized implementation.
* Practice Problems.
* Coding Assignment.
* Solutions.
* Interview Questions.
* Summary.

---

# Week 1 — Day 6 (Part 1)

# Multiple Linear Regression from First Principles

---

# Learning Objectives

By the end of this lesson, you will:

* Understand why a single feature is often not enough.
* Learn what Multiple Linear Regression is.
* Build intuition for multiple input variables.
* Understand the new hypothesis function.
* Visualize how Linear Regression extends to higher dimensions.
* Prepare for vectorized implementations in later parts.

---

# 1. Story-Based Introduction

Imagine you're buying a house.

Yesterday, our model only considered:

> **House Size**

Suppose the model predicts:

| House Size | Predicted Price |
| ---------- | --------------: |
| 1500 sq ft |        ₹80 lakh |

Now consider two different houses.

| House | Size | Bedrooms | Age      | Price       |
| ----- | ---- | -------- | -------- | ----------- |
| A     | 1500 | 2        | 25 years | ₹80 lakh    |
| B     | 1500 | 4        | 2 years  | ₹1.25 crore |

Both houses have the same size.

Yet their prices are very different.

Why?

Because **size isn't the only factor**.

Real house prices depend on many characteristics.

---

# 2. Why Was Multiple Linear Regression Invented?

Single-variable Linear Regression assumes:

> One feature explains everything.

Reality is more complicated.

House prices depend on:

* Size
* Number of bedrooms
* Age
* Distance from the city
* School quality
* Parking
* Neighborhood
* Crime rate
* Public transport
* And many more...

If we ignore these factors, our predictions will often be inaccurate.

Multiple Linear Regression allows the model to learn from **many features at the same time**.

---

# 3. What Is a Feature?

A **feature** is any measurable property of an example that might help predict the target.

Example:

| Feature          | Symbol |
| ---------------- | ------ |
| House Size       | $x_1$  |
| Bedrooms         | $x_2$  |
| Age              | $x_3$  |
| Distance to City | $x_4$  |

Notice the notation.

Instead of one variable $x$, we now have:

$$
x_1,;x_2,;x_3,;\dots,;x_n
$$

where (n) is the number of features.

---

# 4. From One Feature to Many

Yesterday:

$$
\hat{y}=wx+b
$$

Today:

$$
\boxed{
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
}
$$

This is the **Multiple Linear Regression hypothesis function**.

Each feature has its own weight because not every feature contributes equally.

---

# 5. Understanding Every Symbol

In the equation:

$$
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

| Symbol    | Meaning                      |
| --------- | ---------------------------- |
| $\hat{y}$ | Predicted value              |
| $x_1$     | Feature 1 (e.g., house size) |
| $x_2$     | Feature 2 (e.g., bedrooms)   |
| $x_3$     | Feature 3 (e.g., age)        |
| $w_1$     | Weight for feature 1         |
| $w_2$     | Weight for feature 2         |
| $w_3$     | Weight for feature 3         |
| $b$       | Bias (intercept)             |

Each weight tells us **how strongly that feature influences the prediction**.

---

# 6. Why Does Every Feature Need Its Own Weight?

Imagine evaluating a student's final score.

You consider:

* Homework
* Midterm
* Final Exam

Should all contribute equally?

Probably not.

Perhaps:

* Homework → 20%
* Midterm → 30%
* Final Exam → 50%

These percentages are similar to the weights in Multiple Linear Regression.

Some features matter more than others.

The model learns these weights from data.

---

# 7. Worked Example

Suppose our model is:

$$
\hat{y}=50x_1+10x_2-2x_3+30
$$

Where:

* $x_1$ = house size (in hundreds of square feet)
* $x_2$ = number of bedrooms
* $x_3$ = age of the house (years)

Now consider:

| Feature | Value |
| ------- | ----: |
| $x_1$   |    15 |
| $x_2$   |     3 |
| $x_3$   |    10 |

Prediction:

$$
\hat{y} =

50(15)
+
10(3)
-

2(10)
+
30
$$

$$
=750+30-20+30
$$

$$

=790
$$

The model combines the contribution of each feature to produce a single prediction.

---

# 8. Geometry: Line → Plane → Hyperplane

With one feature:

We fit a **line**.

```text
Price
 ^
 |       *
 |    *
 | *
 +------------>
      Size
```

---

With two features:

We fit a **plane**.

```text
          Price
            ^
           /|
          / |
         /  |
        /___|
```

---

With three or more features:

We fit a **hyperplane**.

We can't easily draw a hyperplane because it exists in four or more dimensions, but mathematically it's just the higher-dimensional extension of a line and a plane.

---

# 9. Matrix Intuition

Writing long equations quickly becomes inconvenient.

Instead of:

$$
w_1x_1+w_2x_2+w_3x_3+w_4x_4
$$

we group the features into a vector:

$$
\mathbf{x}=\begin{bmatrix}
x_1\\
x_2\\
x_3\\
x_4
\end{bmatrix}
$$

and the weights into another vector:

$$
\mathbf{w}=
\begin{bmatrix}
w_1\\
w_2\\
w_3\\
w_4
\end{bmatrix}
$$

The hypothesis becomes:

$$
\boxed{\hat{y} = \mathbf{w}^T\mathbf{x}+b}
$$

This is simply the **dot product** you learned on **Day 2**.

Notice how everything in the course is beginning to connect.

---

# 10. Why We Learned Dot Products Earlier

You might have wondered on Day 2:

> "Why are we spending so much time on vectors and dot products?"

Now you know.

Every prediction made by a linear model is essentially a **dot product** between:

* the weight vector
* the feature vector

followed by adding the bias.

This is one of the reasons linear algebra is so important in machine learning.

---

# 11. Feynman Explanation

Imagine trying to predict a student's success.

Would you only look at:

* Hours studied?

No.

You'd also consider:

* Attendance
* Previous grades
* Assignment scores
* Participation

Multiple Linear Regression works exactly the same way.

Instead of relying on one clue, it combines **many clues** to make a better prediction.

---

# 12. Common Mistakes

### ❌ Mistake 1

Assuming more features always improve the model.

Irrelevant or noisy features can actually reduce performance.

---

### ❌ Mistake 2

Thinking every feature has equal importance.

The model learns different weights because different features contribute differently.

---

### ❌ Mistake 3

Confusing the number of features with the number of training examples.

* **Training examples (m):** Number of rows in your dataset.
* **Features (n):** Number of input variables (columns, excluding the target).

For example:

| Houses (rows) | Features (columns) |
| ------------: | -----------------: |
|          1000 |                  4 |

Here:

* (m = 1000)
* (n = 4)

---

# 13. Key Takeaways

* Multiple Linear Regression extends Linear Regression to **multiple input features**.
* Each feature has its own weight.
* The hypothesis is:

$$
\hat{y}=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

* In vector form:

$$
\hat{y}=\mathbf{w}^T\mathbf{x}+b
$$

* The prediction is essentially a **dot product**, connecting directly to what you learned on Day 2.

---

## Reflection

Before moving to Part 2, think about these questions:

1. Why is one feature often insufficient for real-world prediction tasks?
2. Why does each feature need its own weight instead of sharing a single weight?
3. How does the dot product simplify the Multiple Linear Regression equation?

---

## Next: Day 6 — Part 2

We'll extend everything you've learned about Gradient Descent to **multiple features** by covering:

* Cost function for Multiple Linear Regression.
* Partial derivatives for many weights.
* Gradient updates for all parameters.
* Python implementation (before vectorization).
* Intuition for why optimization becomes more challenging as the number of features grows.

This is where your knowledge of **Linear Algebra**, **Calculus**, and **Gradient Descent** comes together into a unified machine learning algorithm.


Excellent. This is where the mathematics from **Day 3 (Linear Regression)**, **Day 4 (Calculus)**, and **Day 5 (Gradient Descent)** finally come together for models with multiple features.

---

# Week 1 — Day 6 (Part 2)

# Cost Function & Gradient Descent for Multiple Linear Regression

---

# Learning Objectives

By the end of this lesson, you will:

* Extend the cost function to multiple features.
* Understand why we need one gradient for each weight.
* Derive the Gradient Descent update rules.
* Learn the vector form of the equations.
* Implement Multiple Linear Regression from scratch (without full vectorization).
* Build intuition for optimization in higher dimensions.

---

# 1. Story-Based Introduction

Imagine you're trying to improve a recipe.

The recipe has four ingredients:

* Flour
* Sugar
* Butter
* Milk

After baking the cake, someone says:

> "It isn't quite right."

Now comes the question:

Should you change:

* only the flour?
* only the sugar?
* both?
* all four?

Unlike our previous model, which had just **one weight and one bias**, we now have **many weights**, each controlling a different feature.

Machine learning faces the same challenge.

Instead of adjusting one knob, it must learn how to adjust **every knob simultaneously**.

---

# 2. Revisiting the Hypothesis Function

For one feature, we had:

$$
\hat{y} = wx + b
$$

With multiple features:

$$
\boxed{
\hat{y}
=
w_1x_1
+
w_2x_2
+
\cdots
+
w_nx_n
+
b
}
$$

Every prediction depends on:

* all features,
* all weights,
* and the bias.

---

# 3. Does the Cost Function Change?

Surprisingly,

**No.**

The idea remains exactly the same.

We still measure how wrong the predictions are.

The cost function is:

$$
\boxed{
J(\mathbf{w}, b)
=\frac{1}{2m}
\sum_{i=1}^{m}
(\hat{y}_i-y_i)^2
}
$$

The only difference is that the prediction

$$
\hat{y}
$$

now depends on many features instead of just one.

---

# 4. Why Do We Need Multiple Gradients?

Yesterday, we updated:

$$
w
$$

Today we have:

$$
w_1,
w_2,
w_3,
\dots,
w_n
$$

Each weight influences the prediction differently.

Therefore,

each one needs its own derivative.

Instead of one gradient,

we compute

$$
\frac{\partial J}{\partial w_1}
$$

$$
\frac{\partial J}{\partial w_2}
$$

$$
\frac{\partial J}{\partial w_3}
$$

...

and so on.

Think of each gradient as answering:

> "If I change this specific weight slightly, how will the cost change?"

---

# 5. Visual Intuition

For one feature:

The cost looked like a bowl.

```text
Cost
 ^
 |      ●
 |    /   \
 |___/_____\____
        w
```

For two weights:

The bowl becomes a surface.

```text
            Cost
              ^
             / \
           /     \
         /         \
_______/_____________\______
         w₁      w₂
```

For hundreds of weights,

the cost exists in hundreds of dimensions.

We can't draw it,

but mathematically the idea remains the same.

---

# 6. The Weight Update Rule

Every weight gets updated independently.

For the first weight:

$$
w_1
:=
w_1
-
\alpha
\frac{\partial J}{\partial w_1}
$$

Second weight:

$$
w_2
:=
w_2
-
\alpha
\frac{\partial J}{\partial w_2}
$$

Third weight:

$$
w_3
:=
w_3
-
\alpha
\frac{\partial J}{\partial w_3}
$$

...

Bias:

$$
b
:=
b
-

\alpha
\frac{\partial J}{\partial b}
$$

Notice the pattern.

Every parameter follows **exactly the same update rule**.

Only the derivative changes.

---

# 7. The Gradient Formula for Any Weight

For the (j)-th weight:

$$
\boxed{
\frac{\partial J}{\partial w_j}
=
\frac1m
\sum_{i=1}^{m}
(\hat y_i-y_i)x_{ij}
}
$$

Let's decode this notation.

| Symbol         | Meaning                                                         |
| -------------- | --------------------------------------------------------------- |
| $m$            | Number of training examples                                     |
| $i$            | Index of the training example                                   |
| $j$            | Index of the feature                                            |
| $x_{ij}$       | The value of the (j)-th feature for the (i)-th training example |
| $\hat y_i-y_i$ | Prediction error                                                |

This formula says:

> Multiply the prediction error by the corresponding feature value, average across all training examples, and you have the gradient for that weight.

---

# 8. Why Does Each Feature Use Its Own Gradient?

Consider predicting house prices.

Features:

* Size
* Bedrooms
* Age

Suppose the model predicts prices that are consistently too low.

Should we increase:

* the size weight?
* the bedroom weight?
* the age weight?

The answer depends on how each feature contributes to the error.

Each gradient measures the influence of **one feature**, allowing the model to adjust each weight independently.

---

# 9. Worked Example

Suppose our model has two features:

$$
\hat y
=

w_1x_1
+
w_2x_2
+
b
$$

Assume:

* $w_1$=1
* $w_2$=2
* $b$=0

Dataset:

| Size ($x_1$) | Bedrooms ($x_2$) | Actual ($y$) |
| -----------: | ---------------: | -----------: |
|            2 |                1 |            5 |
|            3 |                2 |            8 |

---

### Step 1: Predictions

Example 1:

$$
1(2)+2(1)+0=4
$$

Example 2:

$$
1(3)+2(2)+0=7
$$

Predictions:

| Actual | Predicted |
| ------ | --------- |
| 5      | 4         |
| 8      | 7         |

---

### Step 2: Errors

$$
\hat y-y
$$

| Error |
| ----: |
|    -1 |
|    -1 |

---

### Step 3: Gradient for (w_1)

$$
\frac{(-1)(2)+(-1)(3)}2
= -2.5
$$

---

### Step 4: Gradient for (w_2)

$$
\frac{(-1)(1)+(-1)(2)}2
= -1.5
$$

Notice:

The gradients are different because each feature contributes differently.

---

# 10. Python Implementation (Without Full Vectorization)

```python
import numpy as np

# Two features
X = np.array([
    [2, 1],
    [3, 2]
], dtype=float)

y = np.array([5, 8], dtype=float)

# Parameters
w = np.array([1.0, 2.0])
b = 0.0

learning_rate = 0.1

m = len(X)

# Predictions
y_pred = X @ w + b

# Errors
errors = y_pred - y

# Gradients
dw = np.zeros_like(w)

for j in range(len(w)):
    dw[j] = np.mean(errors * X[:, j])

db = np.mean(errors)

# Update
w = w - learning_rate * dw
b = b - learning_rate * db

print("Updated weights:", w)
print("Updated bias:", b)
```

---

# 11. Understanding the Python Code

### Prediction

```python
y_pred = X @ w + b
```

The `@` operator performs **matrix multiplication** (dot product).

For every training example:

$$
\hat y
=
\mathbf{w}^T\mathbf{x}
+
b
$$

This is much cleaner than manually writing:

```python
w1*x1 + w2*x2 + ...
```

---

### Gradient Loop

```python
for j in range(len(w)):
```

This loops through each feature.

For every weight,

we compute one gradient.

Later, in Part 5, we'll remove this loop entirely using **vectorization**.

---

# 12. Why This Isn't Fully Vectorized Yet

Although we're using NumPy arrays, we still have a Python loop over the features.

That's fine for learning.

However, when working with:

* thousands of features,
* millions of training examples,

we want NumPy to perform **all gradient calculations at once**.

We'll see that elegant vectorized solution in Part 5.

---

# 13. Feynman Explanation

Imagine you're tuning a guitar.

It has six strings.

If one string sounds wrong,

you don't tighten every string equally.

You adjust **each string independently**.

Multiple Linear Regression works the same way.

Each weight is like a guitar string.

Each gradient tells you exactly how to tune that one weight.

---

# 14. Common Mistakes

### ❌ Mistake 1

Using the same gradient for every weight.

Each feature has its own gradient because each influences the prediction differently.

---

### ❌ Mistake 2

Confusing the number of features with the number of training examples.

Remember:

* **Rows** = training examples ((m))
* **Columns** = features ((n))

---

### ❌ Mistake 3

Trying to memorize the gradient formulas.

Instead, remember the intuition:

> "Every weight is adjusted according to how much its feature contributes to the prediction error."

---

# Key Takeaways

* The cost function is unchanged in spirit; only the hypothesis becomes more complex.
* Every feature has its own weight and therefore its own gradient.
* Each weight is updated independently using Gradient Descent.
* Matrix multiplication (`@`) naturally extends Linear Regression to multiple features.
* Understanding this loop makes the vectorized implementation much easier.

---

## Reflection

Before moving to Part 3, think about these questions:

1. Why can't all weights share the same gradient?
2. Why does each feature contribute differently to the prediction?
3. Why does using matrix multiplication make the code cleaner as the number of features grows?

---

# Next: Day 6 — Part 3

This is one of the **most practical lessons in the entire Machine Learning course**.

You'll learn **Feature Scaling**, including:

* Why Gradient Descent becomes painfully slow without scaling.
* Why features with different units can dominate learning.
* Visual intuition using elongated cost surfaces.
* When scaling is essential and when it's unnecessary.
* Common scaling techniques used in industry.

This topic explains why many beginners struggle to train models—and how a simple preprocessing step can dramatically improve optimization.


Excellent. This is one of the **most important practical topics** in all of Machine Learning.

Many beginners learn Linear Regression and Gradient Descent correctly, yet their models train **very slowly** or **fail to converge** simply because they skip feature scaling.

If you understand this lesson deeply, you'll avoid one of the most common mistakes in machine learning.

---

# Week 1 — Day 6 (Part 3)

# Feature Scaling from First Principles

---

# Learning Objectives

By the end of this lesson, you will:

* Understand why feature scaling is necessary.
* Learn how features with different units affect Gradient Descent.
* Build intuition using geometric visualizations.
* Understand when feature scaling is required.
* Learn different scaling approaches.
* Prepare for Standardization and Normalization in Part 4.

---

# 1. Story-Based Introduction

Imagine you're moving a sofa.

You're working with a friend.

You push with:

* **50 N** of force.

Your friend pushes with:

* **5000 N** of force.

Who controls the movement?

Obviously,

your friend.

Your effort becomes almost insignificant.

Machine Learning experiences exactly the same problem.

Some features have much larger numerical values than others.

Those large-valued features dominate the optimization process.

---

# 2. A Real Dataset

Suppose we're predicting house prices.

| Feature      | Value |
| ------------ | ----: |
| Size (sq ft) |  2500 |
| Bedrooms     |     3 |
| Age (years)  |    20 |

Notice the scales.

```text
Size      = 2500

Bedrooms = 3

Age       = 20
```

One feature is in the thousands.

Another is a single digit.

Another is in the tens.

These numbers are **not comparable**.

---

# 3. Why Is This a Problem?

Recall the hypothesis:

$$
\hat y
=
w_1x_1
+
w_2x_2
+
w_3x_3
+
b
$$

Suppose

```text
Weight = 1
```

Then

```text
Size contribution

2500 × 1 = 2500
```

Bedrooms

```text
3 × 1 = 3
```

Age

```text
20 × 1 = 20
```

The **Size** feature completely dominates the prediction.

Even if bedrooms are important,

their numerical scale makes their contribution much smaller.

---

# 4. But Can't Gradient Descent Just Learn Smaller Weights?

This is an excellent question.

Yes, **eventually** it can.

But the journey becomes much harder.

Remember,

Gradient Descent doesn't magically know the correct weights.

It has to discover them through many updates.

If one feature is thousands of times larger than another,

the optimization landscape becomes distorted.

This makes learning much slower.

---

# 5. Visual Intuition

Suppose both features have similar scales.

The cost surface looks like a nice round bowl.

```text
          ●

      /       \

    /           \

___/_____________\____
```

Gradient Descent moves smoothly.

---

Now suppose one feature is much larger.

The bowl becomes stretched.

```text
 ______________________

/                      \

\                      /

 \____________________/
```

Instead of moving directly to the minimum,

Gradient Descent zigzags.

---

# 6. Why Does Zigzagging Happen?

Imagine walking down a narrow valley.

```text
\
 \
  \
   ●

  /

 /

/
```

Every update overshoots from one side to the other.

Instead of moving straight toward the minimum,

the algorithm keeps bouncing.

This means:

* More iterations
* More computation
* Slower learning

---

# 7. Another Analogy

Imagine driving.

One road is:

```text
100 km
```

Another is

```text
100 meters
```

If you treat both as if they use the same units,

your navigation becomes completely wrong.

Feature scaling simply ensures that **all features are measured on comparable scales** before training begins.

---

# 8. What Is Feature Scaling?

### Definition

Feature Scaling is the process of transforming numerical features so that they have **similar ranges or magnitudes**.

Instead of:

| Feature  | Original |
| -------- | -------: |
| Size     |     2500 |
| Bedrooms |        3 |
| Age      |       20 |

We transform them into values like:

| Feature  | Scaled |
| -------- | -----: |
| Size     |   0.62 |
| Bedrooms |   0.54 |
| Age      |   0.41 |

Notice:

The relative information remains,

but the scales become comparable.

---

# 9. Does Scaling Change the Data?

No.

This is a common misconception.

Scaling changes:

* the **representation** of the data,

not

* the **information** contained in the data.

Think about measuring height.

Suppose someone is:

```text
180 cm
```

or

```text
1.8 meters
```

The numbers differ,

but the person's height hasn't changed.

Scaling works similarly.

---

# 10. Why Does Gradient Descent Become Faster?

Without scaling,

one gradient may be huge,

another tiny.

Gradient Descent struggles to choose a learning rate that works well for all features.

After scaling,

the gradients become more balanced.

Updates become more uniform,

allowing Gradient Descent to move more directly toward the minimum.

---

# 11. Which Algorithms Need Feature Scaling?

This is a common interview question.

### Usually Required

* Linear Regression (Gradient Descent version)
* Logistic Regression
* Neural Networks
* Support Vector Machines (SVM)
* K-Means Clustering
* K-Nearest Neighbors (KNN)
* Principal Component Analysis (PCA)

These algorithms rely on optimization or distance calculations, so differing feature scales can significantly affect their behavior.

---

### Usually Not Required

* Decision Trees
* Random Forests
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

Tree-based algorithms split based on feature thresholds rather than distances or gradients, so they are generally insensitive to feature scaling.

---

# 12. Why Don't Trees Need Scaling?

Suppose a decision tree asks:

```text
Is House Size > 2000?
```

Whether the feature is stored as:

```text
2000
```

or

```text
2.0
```

the algorithm simply adjusts the threshold.

The ordering of the values stays the same.

Trees care about **relative ordering**, not absolute magnitude.

---

# 13. Common Feature Scaling Techniques

There are many techniques.

The two most important are:

1. **Normalization (Min-Max Scaling)**

Maps values into a fixed range, usually:

$$
[0,1]
$$

---

2. **Standardization (Z-score Scaling)**

Transforms features so that:

* Mean = 0
* Standard Deviation = 1

We'll study both in detail in Part 4.

---

# 14. Feynman Explanation

Imagine three students taking different exams.

Student A:

```text
Math

Marks = 92 /100
```

Student B:

```text
Physics

Marks = 184 /200
```

Student C:

```text
Chemistry

Marks = 46 /50
```

Looking only at the raw scores:

```text
184 > 92 > 46
```

Does that mean Student B performed best?

Not necessarily.

The exams have different maximum scores.

To compare fairly, you first convert all scores to a common scale, such as percentages.

Feature scaling does the same thing for machine learning features.

---

# 15. Common Mistakes

### ❌ Mistake 1

Believing that scaling changes the meaning of the data.

It changes only the numerical representation.

---

### ❌ Mistake 2

Scaling the target variable unnecessarily.

For basic Linear Regression,

we usually scale the **features**, not the target.

(Some advanced workflows may also scale targets, but that's a separate topic.)

---

### ❌ Mistake 3

Assuming every algorithm requires scaling.

Tree-based methods generally do not.

---

### ❌ Mistake 4

Thinking scaling alone improves the quality of the data.

Scaling makes optimization easier,

but it cannot fix poor-quality data or irrelevant features.

---

# 16. Practice Problems

## Easy

1. What is feature scaling?

2. Why does Gradient Descent benefit from scaled features?

3. Does feature scaling change the information in the data?

4. Name two algorithms that usually require feature scaling.

---

## Medium

Suppose a dataset has:

| Feature    | Values    |
| ---------- | --------- |
| House Size | 1000–5000 |
| Bedrooms   | 1–5       |

Explain why Gradient Descent may struggle if these features are not scaled.

---

Suppose someone says:

> "Feature scaling always improves model accuracy."

Do you agree? Explain.

---

## Challenge

A Decision Tree performs identically before and after feature scaling.

Explain **why**.

---

# 17. Interview Questions

1. What is feature scaling?

2. Why is it important?

3. Which algorithms require feature scaling?

4. Which algorithms usually do not?

5. Does feature scaling change the information contained in the data?

6. Why does Gradient Descent converge faster after scaling?

---

# 18. Reflection

Think about these questions before continuing:

1. Why does a feature with much larger numerical values dominate Gradient Descent?
2. Why does scaling make the cost surface easier to optimize?
3. Why are distance-based algorithms especially sensitive to feature scales?

If you can answer these without looking back, you've understood the core intuition behind feature scaling.

---

# Next: Day 6 – Part 4

We'll study the **two most important scaling techniques** used in practice:

* **Normalization (Min-Max Scaling)** — mapping values to a fixed range like ([0,1]).
* **Standardization (Z-score Scaling)** — centering data around a mean of 0 with a standard deviation of 1.

You'll learn:

* the mathematical formulas,
* worked numerical examples,
* NumPy implementations from scratch,
* when to use each method,
* and common interview questions comparing the two.

This is one of the most frequently tested preprocessing topics in machine learning interviews.

Excellent. This is one of the most practical preprocessing lessons in Machine Learning. By the end of this part, you'll not only know **how** to scale data, but **why** different scaling techniques exist and **when** to choose each one.

---

# Week 1 — Day 6 (Part 4)

# Standardization & Normalization from First Principles

---

# Learning Objectives

By the end of this lesson, you will:

* Understand the difference between **Normalization** and **Standardization**.
* Learn the mathematics behind both techniques.
* Compute them manually.
* Implement them using NumPy.
* Understand when to use each method.
* Learn common interview questions and practical considerations.

---

# 1. Story-Based Introduction

Imagine two students applying for the same university.

Student A's exam is graded out of **100**.

Student B's exam is graded out of **1000**.

Scores:

| Student | Score |
| ------- | ----: |
| A       |    92 |
| B       |   910 |

Can we directly compare:

```text
910 > 92
```

No.

The exams have different scales.

A fair comparison requires converting both scores to a common scale.

Machine Learning faces exactly the same problem.

Different features often use completely different units.

---

# 2. The Two Most Popular Solutions

There are many scaling methods, but two dominate practical machine learning.

## Method 1

**Normalization (Min-Max Scaling)**

Maps values into a fixed range.

Usually:

$$
[0,1]
$$

---

## Method 2

**Standardization (Z-score Scaling)**

Centers data around zero while scaling its spread.

After transformation:

* Mean = 0
* Standard Deviation = 1

---

# 3. Understanding Normalization

Suppose we have house sizes.

| House | Size |
| ----: | ---: |
|     1 | 1000 |
|     2 | 1500 |
|     3 | 2000 |
|     4 | 2500 |
|     5 | 3000 |

The smallest value is:

$$
1000
$$

The largest value is:

$$
3000
$$

We want every value to lie between:

$$
0
\quad\text{and}\quad
1
$$

---

# 4. Min-Max Formula

The formula is:

$$
\boxed{
x_{\text{scaled}}
=
\frac{x-x_{\min}}
{x_{\max}-x_{\min}}
}
$$

---

# 5. Understanding Every Symbol

| Symbol              | Meaning                       |
| ------------------- | ----------------------------- |
| $x$                 | Original value                |
| $x_{\min}$          | Smallest value in the feature |
| $x_{\max}$          | Largest value in the feature  |
| $x_{\text{scaled}}$ | Normalized value              |

---

# 6. Worked Example

Suppose

$$
x=2000
$$

Minimum:

$$
1000
$$

Maximum:

$$
3000
$$

Substitute into the formula:

$$
\frac{2000-1000}
{3000-1000}
=
\frac{1000}{2000}
=
0.5
$$

So,

```text
2000

↓

0.5
```

---

# 7. Another Example

Suppose

$$
2500
$$

Normalize it.

$$
\frac{2500-1000}
{3000-1000}
=

\frac{1500}{2000}
=
0.75
$$

---

Result:

| Original | Normalized |
| -------: | ---------: |
|     1000 |       0.00 |
|     1500 |       0.25 |
|     2000 |       0.50 |
|     2500 |       0.75 |
|     3000 |       1.00 |

Everything is now between **0** and **1**.

---

# 8. Python Implementation (Normalization)

```python
import numpy as np

X = np.array([1000, 1500, 2000, 2500, 3000])

X_min = np.min(X)
X_max = np.max(X)

X_norm = (X - X_min) / (X_max - X_min)

print(X_norm)
```

Output

```text
[0.
 0.25
 0.50
 0.75
 1.00]
```

---

# 9. Advantages of Normalization

* Easy to understand.
* Produces values within a fixed range.
* Useful for algorithms that use distances.
* Often used in image processing where pixel values are scaled to **0–1**.

---

# 10. Limitation of Normalization

Suppose one house has a size of:

```text
25000 sq ft
```

Everything else is around:

```text
2000
```

That one extremely large value stretches the range.

Most other values become compressed near zero.

Normalization is therefore **sensitive to outliers**.

---

# 11. Standardization

Instead of forcing values into **0–1**,

Standardization asks:

> **How far is each value from the average?**

This makes it more robust when the data contains extreme values.

---

# 12. Mean

Suppose we have:

```text
10

20

30

40

50
```

Average:

$$
\frac{10+20+30+40+50}{5}
=
30
$$

The mean is:

$$
30
$$

---

# 13. Standard Deviation (Intuition)

Mean tells us the **center** of the data.

Standard deviation tells us:

> **How spread out the values are around the mean.**

Small standard deviation:

```text
29

30

31
```

Everything is close together.

Large standard deviation:

```text
5

30

55
```

Values are spread much farther apart.

---

# 14. Standardization Formula

The formula is:

$$
\boxed{
z
=

\frac{x-\mu}
{\sigma}
}
$$

---

# 15. Understanding Every Symbol

| Symbol   | Meaning                      |
| -------- | ---------------------------- |
| $x$      | Original value               |
| $\mu$    | Mean                         |
| $\sigma$ | Standard deviation           |
| $z$      | Standardized value (Z-score) |

---

# 16. Worked Example

Suppose

Mean:

$$
50
$$

Standard deviation:

$$
10
$$

Value:

$$
70
$$

Then

$$
z
=

\frac{70-50}
{10}
=

2
$$

Interpretation:

The value is

**2 standard deviations above the mean.**

---

Suppose

$$
40
$$

Then

$$
\frac{40-50}
{10}
=
-1
$$

Meaning:

One standard deviation below the mean.

---

# 17. Python Implementation (Standardization)

```python
import numpy as np

X = np.array([10, 20, 30, 40, 50])

mean = np.mean(X)
std = np.std(X)

X_std = (X - mean) / std

print(X_std)
```

Output (approximately)

```text
[-1.414
 -0.707
  0.
  0.707
  1.414]
```

Notice:

The transformed values are **not restricted to the range 0–1**.

They are centered around zero.

---

# 18. Normalization vs Standardization

| Property                 | Normalization                          | Standardization        |
| ------------------------ | -------------------------------------- | ---------------------- |
| Formula                  | $\frac{x-x_{\min}}{x_{\max}-x_{\min}}$ | $\frac{x-\mu}{\sigma}$ |
| Typical Range            | 0–1                                    | No fixed range         |
| Uses Mean?               | No                                     | Yes                    |
| Uses Standard Deviation? | No                                     | Yes                    |
| Sensitive to Outliers?   | Yes                                    | Less sensitive         |
| Centers Data at Zero?    | No                                     | Yes                    |

---

# 19. Which One Should You Use?

There isn't a universal answer.

### Use Normalization when:

* The algorithm depends on distances.
* You want a fixed range like **0–1**.
* Pixel values in images.
* Neural networks with bounded input ranges (in some cases).

---

### Use Standardization when:

* Features are approximately bell-shaped (normally distributed).
* You're using Gradient Descent.
* Logistic Regression.
* Linear Regression.
* Support Vector Machines.
* Principal Component Analysis.

In practice, **Standardization** is often the default choice for many machine learning workflows.

---

# 20. Feynman Explanation

Imagine comparing the heights of students from different schools.

School A measures height in:

```text
centimeters
```

School B measures height in:

```text
inches
```

Before comparing students,

you first convert both to a common measurement system.

Normalization and Standardization do exactly this for machine learning features.

---

# 21. Common Mistakes

### ❌ Mistake 1

Thinking Normalization and Standardization are the same.

They solve similar problems using different transformations.

---

### ❌ Mistake 2

Assuming standardized values must lie between **0** and **1**.

They can be negative or greater than 1.

---

### ❌ Mistake 3

Forgetting that Min-Max Scaling depends on the minimum and maximum values.

New data outside the original range may produce values outside **0–1** unless handled carefully.

---

### ❌ Mistake 4

Computing scaling parameters separately for training and test data.

During model development:

* Compute the scaling parameters (minimum, maximum, mean, standard deviation) **only on the training set**.
* Apply those same parameters to the validation and test sets.

We'll revisit this when we discuss train/test splits.

---

# 22. Practice Problems

## Easy

1. What is Normalization?

2. What is Standardization?

3. Which technique always maps values into the range **0–1**?

4. Which technique produces features with mean **0**?

---

## Medium

Given:

```text
Minimum = 100

Maximum = 300

Value = 220
```

Normalize the value.

---

Given:

```text
Mean = 50

Standard Deviation = 5

Value = 65
```

Compute the standardized value.

---

## Challenge

A dataset contains a few extremely large outliers.

Which preprocessing technique would generally be a better choice:

* Normalization
* Standardization

Explain your reasoning.

---

# 23. Interview Questions

1. What is feature scaling?

2. What is the difference between Normalization and Standardization?

3. Which technique is more sensitive to outliers?

4. Why is Standardization often preferred for Gradient Descent?

5. Should scaling parameters be computed using the test set?

---

# 24. Solutions to Practice Problems

## Easy

### 1. What is Normalization?

Normalization rescales feature values to a fixed range, usually **0–1**, using the minimum and maximum values of the feature.

---

### 2. What is Standardization?

Standardization transforms a feature so it has a mean of **0** and a standard deviation of **1**.

---

### 3. Which technique always maps values into the range 0–1?

**Normalization (Min-Max Scaling).**

---

### 4. Which technique produces features with mean 0?

**Standardization (Z-score Scaling).**

---

## Medium

### Problem 1

Given:

* Minimum = 100
* Maximum = 300
* Value = 220

Using:

$$
\frac{x-x_{\min}}{x_{\max}-x_{\min}}
$$

we get:

$$
\frac{220-100}{300-100}
=
\frac{120}{200}
=
0.6
$$

**Answer:** 0.6

---

### Problem 2

Given:

* Mean = 50
* Standard Deviation = 5
* Value = 65

Using:

$$
z=\frac{x-\mu}{\sigma}
$$

we get:

$$
\frac{65-50}{5}
=

\frac{15}{5}
=
3
$$

**Answer:** 3

The value is **3 standard deviations above the mean**.

---

## Challenge

**Preferred choice:** Standardization.

**Reason:** Normalization relies on the minimum and maximum values, so a few extreme outliers can stretch the range and compress most of the data into a narrow interval. Standardization is generally less affected because it uses the mean and standard deviation, though it is still not completely immune to outliers.

---

# 25. Summary

Today you learned:

* ✅ What Normalization is.
* ✅ What Standardization is.
* ✅ The mathematical formulas behind both.
* ✅ How to compute them manually.
* ✅ How to implement them in NumPy.
* ✅ The advantages and limitations of each technique.
* ✅ When to use each method in practice.

---

## Next: Day 6 – Part 5 (Final)

We'll bring everything together by:

* Building a complete **Multiple Linear Regression** model from scratch.
* Using **vectorized Gradient Descent**.
* Applying **feature scaling**.
* Training the model end-to-end with NumPy.
* Completing the lesson with final coding assignments, interview questions, and a comprehensive Week 1 Day 6 summary.

By the end of Part 5, you'll have implemented a realistic machine learning pipeline that closely resembles how many classical machine learning models are trained in practice.


Excellent! This is the final part of **Week 1 – Day 6**. We'll now combine everything you've learned over the last six days into a complete machine learning pipeline.

By the end of this lesson, you'll have built a **Multiple Linear Regression model from scratch**, using **feature scaling** and **vectorized Gradient Descent** with only **NumPy**.

---

# Week 1 — Day 6 (Part 5)

# Complete Multiple Linear Regression Pipeline & Wrap-up

---

# Learning Objectives

By the end of this lesson, you will:

* Build a complete Multiple Linear Regression model.
* Apply feature scaling before training.
* Implement fully vectorized Gradient Descent.
* Track training progress.
* Understand why vectorization is preferred.
* Complete Day 6 with practice problems, coding assignments, interview questions, and solutions.

---

# 1. Story-Based Introduction

Imagine you're building a house price prediction system.

Your workflow isn't just:

```text
Data → Prediction
```

A real machine learning pipeline looks more like this:

```text
Collect Data
      │
      ▼
Clean Data
      │
      ▼
Scale Features
      │
      ▼
Train Model
      │
      ▼
Evaluate Model
      │
      ▼
Make Predictions
```

Today, you'll build the heart of that pipeline.

---

# 2. The Dataset

Suppose we have:

| Size | Bedrooms | Age | Price |
| ---: | -------: | --: | ----: |
| 1000 |        2 |  20 |   150 |
| 1500 |        3 |  15 |   220 |
| 2000 |        3 |  10 |   280 |
| 2500 |        4 |   8 |   340 |
| 3000 |        4 |   5 |   400 |

Features:

* Size
* Bedrooms
* Age

Target:

* Price

---

# 3. Step 1 — Create the Dataset

```python
import numpy as np

X = np.array([
    [1000, 2, 20],
    [1500, 3, 15],
    [2000, 3, 10],
    [2500, 4, 8],
    [3000, 4, 5]
], dtype=float)

y = np.array([150,220,280,340,400], dtype=float)
```

---

# 4. Step 2 — Standardize the Features

Using what you learned in Part 4:

```python
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_scaled = (X - mean) / std
```

Notice

```python
axis=0
```

means

> Compute statistics **column-wise**.

Each feature gets its own mean and standard deviation.

---

# 5. Step 3 — Initialize Parameters

Suppose there are

```python
n = X.shape[1]
```

features.

Initialize:

```python
w = np.zeros(n)
b = 0.0
```

Instead of:

```python
w1

w2

w3
```

we store all weights inside one vector.

---

# 6. Step 4 — Predictions

Instead of writing

```python
w1*x1+w2*x2+w3*x3
```

use matrix multiplication.

```python
y_pred = X_scaled @ w + b
```

This computes predictions for **all training examples at once**.

---

# 7. Step 5 — Errors

```python
errors = y_pred - y
```

Exactly as before.

Nothing changes.

---

# 8. Step 6 — Compute Cost

```python
cost = np.mean(errors**2)
```

Or

```python
cost = np.sum(errors**2)/(2*len(X))
```

Both are acceptable.

The second matches the traditional Linear Regression cost function.

---

# 9. Step 7 — Fully Vectorized Gradients

Previously we used a loop.

Now NumPy computes all gradients simultaneously.

Weight gradients:

```python
dw = (X_scaled.T @ errors)/len(X)
```

Bias gradient:

```python
db = np.mean(errors)
```

Let's understand this equation.

---

# 10. Why Does This Work?

Recall the formula

$$
\frac{\partial J}{\partial w_j}
===============================

\frac1m
\sum
(\hat y-y)x_j
$$

Instead of computing one feature at a time,

matrix multiplication computes **every feature's gradient simultaneously**.

The transpose

```python
X.T
```

changes the shape so that each feature is aligned with its corresponding errors.

This is one of the biggest advantages of linear algebra in machine learning.

---

# 11. Step 8 — Update Parameters

```python
w -= learning_rate * dw
b -= learning_rate * db
```

Exactly the same update rule.

Now it updates **every weight at once**.

---

# 12. Complete Training Algorithm

```python
import numpy as np

X = np.array([
    [1000,2,20],
    [1500,3,15],
    [2000,3,10],
    [2500,4,8],
    [3000,4,5]
], dtype=float)

y = np.array([150,220,280,340,400], dtype=float)

# Standardization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X = (X - mean)/std

m, n = X.shape

w = np.zeros(n)
b = 0

learning_rate = 0.01
epochs = 1000

cost_history = []

for epoch in range(epochs):

    y_pred = X @ w + b

    errors = y_pred - y

    cost = np.sum(errors**2)/(2*m)

    cost_history.append(cost)

    dw = (X.T @ errors)/m

    db = np.mean(errors)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} Cost={cost:.4f}")

print("\nWeights:", w)
print("Bias:", b)
```

---

# 13. Understanding the Shapes (Very Important)

Suppose:

```text
5 houses

3 features
```

Then

| Variable | Shape |
| -------- | ----- |
| X        | (5,3) |
| w        | (3,)  |
| X @ w    | (5,)  |
| y        | (5,)  |
| errors   | (5,)  |
| X.T      | (3,5) |
| dw       | (3,)  |

Understanding shapes is one of the most valuable debugging skills in machine learning.

---

# 14. Why Vectorization Matters

Suppose:

1 million samples

100 features

Using Python loops:

```text
Very Slow
```

Using NumPy:

```text
Extremely Fast
```

Because NumPy performs optimized operations internally rather than looping in Python.

---

# 15. Complete ML Pipeline

```text
Raw Data
    │
    ▼
Feature Scaling
    │
    ▼
Initialize Parameters
    │
    ▼
Predictions
    │
    ▼
Errors
    │
    ▼
Cost Function
    │
    ▼
Gradients
    │
    ▼
Update Parameters
    │
    ▼
Repeat Until Convergence
    │
    ▼
Final Model
```

Notice how every topic you've learned in Week 1 fits into one complete workflow.

---

# 16. Common Mistakes

### ❌ Mistake 1

Forgetting to scale features before Gradient Descent.

Training may become unnecessarily slow or unstable.

---

### ❌ Mistake 2

Mixing up rows and columns.

Remember:

* Rows = training examples.
* Columns = features.

---

### ❌ Mistake 3

Computing gradients with the wrong matrix dimensions.

Always verify the shapes of `X`, `X.T`, `w`, `y_pred`, and `errors`.

---

### ❌ Mistake 4

Using explicit Python loops when vectorized NumPy operations are available.

---

# 17. Practice Problems

## Easy

1. Why is `X.T @ errors` used instead of `X @ errors`?

2. Why do we standardize features before Gradient Descent?

3. What is the shape of `w` if there are 10 features?

4. Why do we store all weights in one NumPy array?

---

## Medium

Suppose:

* 500 samples
* 8 features

Answer:

1. What is the shape of `X`?
2. What is the shape of `X.T`?
3. What is the shape of `w`?
4. What is the shape of `X @ w`?

---

Suppose your cost increases after every epoch.

List at least three possible causes.

---

## Challenge

Explain why matrix multiplication allows NumPy to compute all gradients simultaneously without explicitly looping over each feature.

---

# 18. Coding Assignment

## Task 1

Using the dataset provided in this lesson:

* Train Multiple Linear Regression.
* Print the cost every 100 epochs.
* Store the cost history.
* Print the final weights and bias.

---

## Task 2

After training,

predict the price for a new house:

* Size = 2200
* Bedrooms = 3
* Age = 12

**Important:** Standardize the new house using the **training mean and standard deviation** before making the prediction.

---

## Bonus

Write a reusable function:

```python
train_linear_regression(X, y, learning_rate, epochs)
```

that returns:

```python
weights,
bias,
cost_history
```

This is a great exercise in writing reusable machine learning code.

---

# 19. Interview Questions

1. What is Multiple Linear Regression?

2. Why do we use matrix multiplication?

3. What is vectorization?

4. Why is vectorization faster?

5. Why should scaling parameters be computed only on the training set?

6. What is the shape of the weight vector for (n) features?

7. Why is feature scaling important for Gradient Descent?

8. What does `X.T @ errors` compute?

9. Why do we standardize a new sample before prediction?

10. What is the complete training pipeline for Multiple Linear Regression?

---

# 20. Solutions to Practice Problems

## Easy

### 1. Why is `X.T @ errors` used instead of `X @ errors`?

`X.T` has shape `(n, m)` while `errors` has shape `(m,)`. Their matrix multiplication produces a vector of shape `(n,)`, giving one gradient for each feature. `X @ errors` is not dimensionally compatible in this context.

---

### 2. Why do we standardize features before Gradient Descent?

Standardization puts features on comparable scales, making Gradient Descent more stable and allowing it to converge faster.

---

### 3. What is the shape of `w` if there are 10 features?

```text
(10,)
```

One weight for each feature.

---

### 4. Why do we store all weights in one NumPy array?

It enables vectorized operations, making the code cleaner, faster, and scalable to many features.

---

## Medium

Given:

* 500 samples
* 8 features

Shapes:

* `X` → `(500, 8)`
* `X.T` → `(8, 500)`
* `w` → `(8,)`
* `X @ w` → `(500,)`

---

If the cost increases after every epoch, possible causes include:

* Learning rate is too large.
* Gradient calculations are incorrect.
* Parameter update equations have the wrong sign.
* Data contains extreme values and features haven't been scaled.

---

## Challenge

Matrix multiplication combines the contributions of all features across all training examples in one optimized operation. Instead of calculating each feature's gradient in a Python loop, linear algebra performs the entire computation simultaneously, making the implementation both simpler and significantly faster.

---

# 21. Solution to Coding Assignment

```python
import numpy as np

# Data
X = np.array([
    [1000,2,20],
    [1500,3,15],
    [2000,3,10],
    [2500,4,8],
    [3000,4,5]
], dtype=float)

y = np.array([150,220,280,340,400], dtype=float)

# Scale training data
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_scaled = (X - mean) / std

m, n = X_scaled.shape

w = np.zeros(n)
b = 0.0

learning_rate = 0.01
epochs = 1000

cost_history = []

for epoch in range(epochs):
    y_pred = X_scaled @ w + b
    errors = y_pred - y

    cost = np.sum(errors**2) / (2 * m)
    cost_history.append(cost)

    dw = (X_scaled.T @ errors) / m
    db = np.mean(errors)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Cost = {cost:.4f}")

print("\nFinal Weights:", w)
print("Final Bias:", b)

# Predict a new house
new_house = np.array([2200, 3, 12], dtype=float)
new_house_scaled = (new_house - mean) / std

prediction = new_house_scaled @ w + b
print(f"\nPredicted Price: {prediction:.2f}")
```

---

# 22. Week 1 – Day 6 Summary

Today you learned:

* ✅ Multiple Linear Regression with multiple features.
* ✅ Cost functions and Gradient Descent for multiple parameters.
* ✅ Why feature scaling is critical.
* ✅ The difference between Normalization and Standardization.
* ✅ How to implement scaling with NumPy.
* ✅ How to build a fully vectorized Multiple Linear Regression model.
* ✅ How matrix multiplication simplifies machine learning computations.
* ✅ How all the concepts from Week 1 fit together into a complete training pipeline.

---

# 🎉 Week 1 Complete!

You have now completed all the **new concepts** for Week 1:

* **Day 1:** Introduction to AI, ML & Deep Learning
* **Day 2:** Python, NumPy, Vectors, Matrices & Dot Products
* **Day 3:** Linear Regression
* **Day 4:** Calculus for Machine Learning
* **Day 5:** Gradient Descent
* **Day 6:** Multiple Linear Regression & Feature Scaling

These six days form the mathematical and programming foundation for the rest of the course.

## Up Next: Week 1 – Day 7

Following your curriculum, Day 7 is **not** a new theory day. It is a consolidation day focused on applying what you've learned through:

* Comprehensive revision of Week 1.
* A complete **House Price Prediction Project** built from scratch (without Scikit-Learn).
* A quiz.
* Reflection exercises.
* Interview preparation.
* A practical coding challenge that integrates all of Week 1's concepts.

This project will reinforce everything you've learned before moving into **Week 2: Classification & Data Preparation**.
