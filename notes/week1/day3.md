# Week 1 — Day 3 (Part 1)

## Linear Regression from First Principles

---

# 1. Learning Objectives

By the end of this lesson, you will be able to:

* Explain what Linear Regression is.
* Understand why it was invented.
* Define the hypothesis (prediction) function.
* Understand the relationship between features (X) and targets (y).
* Build a simple Linear Regression model from intuition before learning the mathematics.

---

# 2. Story-Based Introduction

Imagine you've just started a small real estate company.

A customer walks into your office and asks:

> "How much should I pay for a 1,500 sq. ft. house?"

You don't know.

But you have records of houses sold previously.

| House Size (sq ft) | Price ($1000) |
| -----------------: | ------------: |
|                800 |           150 |
|               1000 |           180 |
|               1200 |           210 |
|               1500 |           260 |
|               1800 |           300 |

You notice something interesting:

* Bigger houses usually cost more.
* Smaller houses usually cost less.

If you draw these points on a graph, they almost form a straight line.

```
Price
 ^
 |                         *
 |                    *
 |               *
 |          *
 |     *
 +------------------------------------>
            House Size
```

Now a customer asks about a **1600 sq ft** house.

There is no previous record.

How can we estimate its price?

This is the exact problem **Linear Regression** was created to solve.

---

# 3. Why Was Linear Regression Invented?

Before jumping into formulas, let's answer the most important question:

## The Problem

We often know **past observations**.

But we need to predict **future values**.

Examples:

* Predict house prices
* Predict salaries
* Predict rainfall
* Predict fuel consumption
* Predict electricity demand
* Predict company revenue

Humans can estimate by intuition.

Computers cannot.

They need a mathematical rule.

Linear Regression is one of the simplest ways to learn that rule.

---

# 4. What Would Happen Without Linear Regression?

Suppose we have 10,000 houses.

Whenever a new customer arrives, we'd have to:

* Search all previous houses
* Compare sizes
* Estimate manually
* Hope we're close

This becomes slow, inconsistent, and doesn't scale.

Instead, we want the computer to learn a simple equation like:

```
Price = Some Formula(Size)
```

Then predicting becomes instant.

---

# 5. What Is Linear Regression?

**Definition:**

Linear Regression is an algorithm that learns the relationship between input variables (features) and a continuous output (target) by fitting the best possible straight line through the data.

The keyword here is **continuous**.

Examples of continuous values:

* ₹45.25
* 182.6 cm
* 73.4 kg
* 98.7°F

Not categories like:

* Cat/Dog
* Spam/Not Spam
* Healthy/Sick

Those are classification problems, which we'll study in Week 2.

---

# 6. Intuition

Imagine plotting every house sale on a graph.

```
Price
 ^
 |                       *
 |                  *
 |             *
 |        *
 |   *
 +------------------------------------>
        House Size
```

There are infinitely many lines you could draw.

```
Too Steep

     /
    /
   /
  /

Too Flat

------------

Reasonable

      /
     /
    /
```

Our goal is to find **the line that best represents the trend in the data**, not necessarily one that passes through every point.

This is because real-world data contains noise:

* A small house might be expensive because it's in a prime location.
* A large house might be cheap because it's old.

The model should capture the overall pattern, not every exception.

---

# 7. Real-World Analogy

Imagine throwing a handful of magnets onto a table.

They're scattered.

Now place a ruler through the middle so that it's as close as possible to all the magnets.

The ruler won't touch every magnet.

But it represents the overall direction.

Linear Regression does exactly this with data.

---

# 8. Understanding Features (X) and Target (y)

Every supervised learning problem has two parts.

### Feature (Input)

Information we already know.

Examples:

* House size
* Years of experience
* Hours studied
* Engine size

We denote features by:

```
X
```

### Target (Output)

The value we want to predict.

Examples:

* House price
* Salary
* Exam score
* Fuel consumption

We denote the target by:

```
y
```

Example:

| House Size (X) | Price (y) |
| -------------: | --------: |
|            800 |       150 |
|           1000 |       180 |
|           1200 |       210 |
|           1500 |       260 |

Think of it like a function in mathematics:

```
Input  →  Function  → Output

X       →    f()    →   y
```

Machine learning is essentially about discovering what the function **f()** should be.

---

# 9. The Hypothesis Function

In machine learning, the model's prediction is called the **hypothesis**.

We write it as:

[
\hat{y} = h(x)
]

Read as:

> "y-hat equals h of x."

Here:

* (x) = input feature
* (h(x)) = prediction made by the model
* (\hat{y}) ("y-hat") = predicted value
* (y) = actual value

The hat (^) indicates an estimate rather than the true value.

---

# 10. Our First Prediction Equation

For a straight line, the hypothesis is:

[
\hat{y} = mx + c
]

In machine learning, we usually write it as:

[
\hat{y} = w x + b
]

Where:

* **w** = weight (slope of the line)
* **b** = bias (intercept, where the line crosses the y-axis)
* **x** = input feature
* **ŷ** = predicted output

If:

```
w = 2
b = 5
```

Then:

```
ŷ = 2x + 5
```

Example predictions:

|  x | Prediction (ŷ) |
| -: | -------------: |
|  1 |              7 |
|  2 |              9 |
|  3 |             11 |
|  4 |             13 |

Every increase of 1 in **x** increases the prediction by **2**, because the slope (**w**) is 2.

---

# 11. Feynman Explanation

Imagine you're teaching a 10-year-old:

> "We have many examples of how big houses are and how much they cost. We notice that bigger houses usually cost more. Instead of memorizing every house, we draw one straight line that captures this pattern. Whenever someone tells us the size of a new house, we look at the line to estimate its price."

That's Linear Regression in its simplest form.

---

## Key Takeaways from Part 1

* Linear Regression predicts **continuous numerical values**.
* It learns the relationship between **features (X)** and **target (y)**.
* The model's prediction is called the **hypothesis**.
* The hypothesis for a straight line is:

[
\hat{y} = wx + b
]

* The challenge now is to determine the **best values of (w) and (b)**.

In **Part 2**, we'll answer the next critical question:

> **How does a machine know whether its line is good or bad?**

This leads us to the **Cost Function** and **Mean Squared Error (MSE)**, which are the foundation of training almost every machine learning model.

# Week 1 — Day 3 (Part 2)

# Cost Function & Mean Squared Error (MSE)

---

# 1. Learning Objectives

By the end of this part, you will:

* Understand why we need a Cost Function.
* Learn what an error (or residual) is.
* Understand why simply adding errors doesn't work.
* Derive the Mean Squared Error (MSE) intuitively.
* Understand why squaring errors is important.
* Learn the mathematical formula for the cost function.

---

# 2. Story-Based Introduction

Let's go back to our real estate company.

Suppose your first prediction model is:

[
\hat{y} = 0.1x + 80
]

Now let's test it.

| House Size | Actual Price | Predicted Price |
| ---------: | -----------: | --------------: |
|       1000 |          180 |             180 |
|       1200 |          210 |             200 |
|       1500 |          260 |             230 |

The first prediction is perfect.

The second is off by **10**.

The third is off by **30**.

A natural question arises:

> **How do we measure whether our model is good or bad?**

Without a way to measure mistakes, the computer has no idea how to improve.

---

# 3. Why Was the Cost Function Invented?

Imagine you're learning to play basketball.

Every shot can either:

* Go in 🏀
* Miss ❌

If no one tells you whether your shots are good or bad, how will you improve?

Machine learning faces the same problem.

The model needs **feedback**.

The Cost Function is that feedback.

Think of it as a **scoreboard**:

* Small cost → Good model
* Large cost → Poor model

The goal of training is simply:

> **Find the model with the smallest possible cost.**

---

# 4. What Is an Error?

Every prediction has two values:

* Actual value ((y))
* Predicted value ((\hat{y}))

The difference is called the **error** (also called the **residual**).

[
\text{Error} = y - \hat{y}
]

Example:

| Actual | Predicted | Error |
| -----: | --------: | ----: |
|    200 |       190 |    10 |
|    150 |       160 |   -10 |
|    300 |       290 |    10 |

Positive error means we predicted **too low**.

Negative error means we predicted **too high**.

---

# 5. Why Can't We Just Add the Errors?

Suppose our predictions are:

| Error |
| ----: |
|   +20 |
|   -20 |

Adding them gives:

[
20 + (-20) = 0
]

Does that mean the model is perfect?

Of course not!

Both predictions were wrong by **20**.

The mistakes simply canceled each other out.

```text
Prediction 1  +20

Prediction 2  -20

Total          0 ❌
```

So simply adding errors doesn't work.

---

# 6. First Attempt: Use Absolute Values

One idea is:

[
|y-\hat{y}|
]

Now:

| Error | Absolute Error |
| ----: | -------------: |
|   +20 |             20 |
|   -20 |             20 |

No cancellation.

Great!

But there's a problem.

Absolute values are harder to optimize mathematically because they are **not differentiable at zero**. Since we'll use derivatives in Gradient Descent (Day 5), we prefer a smoother function.

So while **Mean Absolute Error (MAE)** is useful, Linear Regression is traditionally trained with **Mean Squared Error (MSE)**.

---

# 7. The Brilliant Idea: Square the Errors

Instead of taking the absolute value, square the error.

[
(y-\hat{y})^2
]

Example:

| Error | Squared Error |
| ----: | ------------: |
|     5 |            25 |
|    -5 |            25 |
|    20 |           400 |
|   -20 |           400 |

Notice two things:

1. All values become positive.
2. Larger mistakes become much more expensive.

---

# 8. Why Punish Large Errors More?

Imagine two students.

### Student A

Misses every question by **1 mark**.

Errors:

```text
1
1
1
1
```

### Student B

Gets most correct but one answer is wildly wrong.

Errors:

```text
0
0
0
10
```

Without squaring:

```text
Student A = 4

Student B = 10
```

With squaring:

```text
Student A

1²+1²+1²+1² = 4

Student B

10² = 100
```

Student B is penalized much more because one huge mistake can be far more harmful in many applications.

This is why squaring is often preferred.

---

# 9. Visual Intuition

Imagine a bowl.

```text
             Cost

              ^
             / \
            /   \
           /     \
          /       \
---------/---------\------------>

            Weight
```

The bottom of the bowl is the **lowest cost**.

Training a model is like rolling a ball to the bottom.

We'll see on Day 5 that **Gradient Descent** is the algorithm that helps the ball roll downhill.

---

# 10. Mean Squared Error (MSE)

Suppose we have (n) training examples.

For each example:

1. Compute the prediction.
2. Find the error.
3. Square it.
4. Add all squared errors.
5. Divide by the number of examples.

This gives the **Mean Squared Error (MSE)**:

[
\boxed{\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
]

---

# 11. Understanding Every Symbol

Let's break the formula down:

[
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
]

| Symbol              | Meaning                               |
| ------------------- | ------------------------------------- |
| (n)                 | Number of training examples           |
| (\sum)              | Sum all values                        |
| (i)                 | Index of each training example        |
| (y_i)               | Actual value of the (i)-th example    |
| (\hat{y}_i)         | Predicted value of the (i)-th example |
| ((y_i-\hat{y}_i)^2) | Squared error for one example         |

The **Mean** means average, and **Squared Error** means each error is squared before averaging.

---

# 12. The Cost Function for Linear Regression

In machine learning, we usually denote the cost function by:

[
J(w,b)
]

This reminds us that the cost depends on the parameters **(w)** (weight) and **(b)** (bias).

For Linear Regression, the cost function is:

[
\boxed{
J(w,b)=\frac{1}{2n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}
]

You might notice something new:

[
\frac{1}{2n}
]

instead of

[
\frac{1}{n}
]

Why divide by **2**?

It **does not change which model is best**. It is added purely because, when we differentiate this function later, the factor of 2 from the square term cancels neatly, making the math cleaner.

You can think of it as a mathematical convenience.

---

# 13. Worked Example

Suppose we have three houses.

| House | Actual ((y)) | Predicted ((\hat{y})) |
| ----: | -----------: | --------------------: |
|     1 |          200 |                   190 |
|     2 |          250 |                   260 |
|     3 |          300 |                   290 |

### Step 1: Calculate Errors

| House | Error |
| ----: | ----: |
|     1 |    10 |
|     2 |   -10 |
|     3 |    10 |

### Step 2: Square Errors

| House | Squared Error |
| ----: | ------------: |
|     1 |           100 |
|     2 |           100 |
|     3 |           100 |

### Step 3: Add Them

[
100+100+100=300
]

### Step 4: Divide by the Number of Examples

[
\text{MSE}=\frac{300}{3}=100
]

If using the Linear Regression cost function:

[
J=\frac{300}{2\times3}=50
]

The absolute value isn't important by itself. What matters is that **lower cost indicates a better model**.

---

# 14. Feynman Explanation

Imagine you're throwing darts at a dartboard.

* The bullseye is the correct answer.
* Every dart is a prediction.
* The farther a dart lands from the bullseye, the bigger the mistake.
* Missing by a lot is worse than missing by a little.

The Cost Function simply measures **how far, on average, all your darts are from the bullseye**.

Your goal is to adjust your aim until this average distance is as small as possible.

---

# 15. Key Takeaways

* Every prediction has an associated error.
* Adding raw errors is misleading because positive and negative errors can cancel out.
* Squaring errors ensures they are always positive and emphasizes larger mistakes.
* **MSE** is the average of all squared errors.
* The **Cost Function** measures how well our model performs.
* Training a Linear Regression model is essentially the process of finding the values of **(w)** and **(b)** that minimize this cost.

---

## Reflection Question

Before moving on, think about this:

> If two models predict the same dataset and one has an MSE of **20** while the other has an MSE of **200**, which model would you choose, and why? Also, what might cause the second model to have a much higher error?

In **Part 3**, we'll make all of this concrete by:

* implementing Linear Regression predictions and MSE from scratch using **Python and NumPy**,
* visualizing how changing **(w)** and **(b)** affects the fitted line,
* building intuition for the **Normal Equation**, and
* finishing the lesson with practice problems, assignments, interview questions, and a summary.


# Week 1 — Day 3 (Part 3)

# Python Implementation, Normal Equation (Intuition), Practice & Wrap-up

Congratulations! You've now understood **why** Linear Regression exists and **how** we measure whether a model is good. Now it's time to implement everything from scratch.

---

# 1. Python Implementation (From Scratch)

We'll use only **Python + NumPy**, as specified in the course curriculum. No Scikit-Learn yet.

## Dataset

```python
import numpy as np

# House size (sq ft)
X = np.array([800, 1000, 1200, 1500, 1800])

# House prices ($1000)
y = np.array([150, 180, 210, 260, 300])
```

Here:

* `X` = Features
* `y` = Actual target values

---

# 2. Define Our Model

Suppose we **guess**:

[
w = 0.15,\qquad b = 20
]

```python
w = 0.15
b = 20
```

Remember:

* `w` → slope
* `b` → intercept

---

# 3. Make Predictions

The hypothesis is:

[
\hat{y}=wx+b
]

In Python:

```python
y_pred = w * X + b

print(y_pred)
```

Output:

```text
[140. 170. 200. 245. 290.]
```

Let's compare.

| House Size | Actual | Predicted |
| ---------- | ------ | --------- |
| 800        | 150    | 140       |
| 1000       | 180    | 170       |
| 1200       | 210    | 200       |
| 1500       | 260    | 245       |
| 1800       | 300    | 290       |

Notice:

The predictions are close, but not perfect.

---

# 4. Compute Errors

```python
errors = y - y_pred

print(errors)
```

Output

```text
[10. 10. 10. 15. 10.]
```

Each value tells us:

> "How far away was the prediction?"

---

# 5. Square the Errors

```python
squared_errors = errors ** 2

print(squared_errors)
```

Output

```text
[100.
100.
100.
225.
100.]
```

Notice that all values are now positive.

---

# 6. Compute Mean Squared Error

```python
mse = np.mean(squared_errors)

print(mse)
```

Output

```text
125.0
```

That means:

Our current model has an average squared error of **125**.

---

# 7. Build Everything Into a Function

A good programmer avoids repeating code.

```python
import numpy as np

def compute_mse(X, y, w, b):

    predictions = w * X + b

    errors = y - predictions

    squared_errors = errors ** 2

    mse = np.mean(squared_errors)

    return mse
```

Now we can test many different models.

```python
cost = compute_mse(X, y, 0.15, 20)

print(cost)
```

Output

```text
125.0
```

---

# 8. Try Different Values

Model 1

```python
print(compute_mse(X, y, 0.15, 20))
```

```
125
```

Model 2

```python
print(compute_mse(X, y, 0.16, 15))
```

Maybe

```
70
```

Model 3

```python
print(compute_mse(X, y, 0.18, 5))
```

Maybe

```
600
```

Which model is better?

The one with the **smallest MSE**.

That's all training is:

> Try to find the values of **w** and **b** that minimize the cost.

---

# 9. Visual Explanation

Imagine trying different lines.

```text
Cost

900 |             ●
800 |
700 |
600 |        ●
500 |
400 |
300 |
200 |    ●
100 | ●
 50 |____________________________

      Different Models
```

Lower point = Better model.

---

# 10. Can We Just Try Every Possible Line?

Suppose:

```
w = 0.000001
```

Then

```
0.000002

0.000003

0.000004
```

There are infinitely many possible values.

Trying every combination is impossible.

So we need a smarter way.

There are **two major approaches**:

### Method 1

Solve the mathematics directly.

This gives the **Normal Equation**.

### Method 2

Start somewhere and gradually improve.

This gives **Gradient Descent**.

We'll study Gradient Descent in detail on **Day 5**.

---

# 11. The Normal Equation (Intuition)

Suppose someone asks:

> "Can we calculate the best line directly without trial and error?"

For Linear Regression, the answer is **yes**.

There is a closed-form mathematical solution called the **Normal Equation**.

It computes the optimal weights in one calculation (for suitable datasets).

The formula is:

[
\boxed{\theta=(X^TX)^{-1}X^Ty}
]

Don't worry if this looks intimidating—we're focusing on intuition today.

Where:

* (X) = Design matrix (all features)
* (X^T) = Transpose of (X)
* ((X^TX)^{-1}) = Matrix inverse
* (y) = Target values
* (\theta) = Parameters (weights and bias)

---

# 12. Why Don't We Always Use the Normal Equation?

It sounds perfect:

* No iterations
* No learning rate
* No epochs

So why isn't it always used?

Because for very large datasets:

* Matrix multiplication becomes expensive.
* Matrix inversion can be slow and memory-intensive.
* Some matrices may not even be invertible.

Gradient Descent scales much better to massive datasets and is also applicable to many models where no closed-form solution exists.

---

# 13. Comparison

| Normal Equation            | Gradient Descent                  |
| -------------------------- | --------------------------------- |
| Exact solution             | Iterative solution                |
| No learning rate           | Requires learning rate            |
| Good for small datasets    | Excellent for very large datasets |
| Matrix inversion required  | No matrix inversion               |
| Only for Linear Regression | Used for many ML models           |

---

# 14. Common Mistakes

### Mistake 1

Using classification for regression problems.

Regression predicts numbers, not categories.

---

### Mistake 2

Thinking a perfect fit is always best.

A model that memorizes the training data may perform poorly on new, unseen data (overfitting).

---

### Mistake 3

Ignoring the units of the target.

Predicting house prices in dollars versus thousands of dollars changes the scale of the MSE.

---

### Mistake 4

Assuming a lower training error always means a better model.

What ultimately matters is how well the model generalizes to new data, which we'll discuss later.

---

# 15. Practice Problems

## Easy

1. What is Linear Regression?

2. What type of output does it predict?

3. What is the difference between (y) and (\hat{y})?

4. Why do we square the errors?

---

## Medium

5. A model predicts:

```
Actual      Predicted

100         90

120         130

150         145
```

Calculate:

* Errors
* Squared errors
* MSE

---

6. If the MSE decreases from 150 to 20, what does that indicate?

---

## Challenge

7. Explain why using only the sum of errors can be misleading.

8. Compare MSE and MAE. When might one be preferred over the other?

---

# 16. Coding Assignment

## Task

Write a function:

```python
linear_regression_predict(X, w, b)
```

that returns:

[
\hat{y}=wX+b
]

Then write another function:

```python
compute_mse(y_true, y_pred)
```

Finally:

* Test at least **three** different values of `w` and `b`.
* Determine which combination yields the lowest MSE.

**Bonus:** Print the results in a table.

---

# 17. Interview Questions

1. What is Linear Regression?

2. Why is it called "linear"?

3. What is the hypothesis function?

4. What is the Cost Function?

5. Why do we square the errors?

6. What is Mean Squared Error?

7. What is the difference between MSE and MAE?

8. Why do we divide by (2n) in the Linear Regression cost function?

9. What is the Normal Equation?

10. When would you choose Gradient Descent over the Normal Equation?

---

# 18. Reflection

Take a moment to reflect on these questions:

* Why can't a machine simply "guess" the correct line?
* If two different lines fit the data, how should the machine decide which one is better?
* Why is a numerical measure of error essential for learning?

If you can explain these ideas in your own words, you've understood the core concept behind supervised learning.

---

# 19. Solutions to Practice Problems

### Easy

1. **Linear Regression** is an algorithm that predicts continuous numerical values by learning the best-fitting straight-line relationship between inputs and outputs.

2. It predicts **continuous values**, such as prices, temperatures, or salaries.

3. (y) is the **actual** value, while (\hat{y}) is the **predicted** value.

4. We square errors to make them positive and to penalize larger mistakes more heavily.

---

### Medium (Problem 5)

| Actual | Predicted | Error ((y-\hat{y})) | Squared Error |
| ------ | --------- | ------------------- | ------------- |
| 100    | 90        | 10                  | 100           |
| 120    | 130       | -10                 | 100           |
| 150    | 145       | 5                   | 25            |

Total squared error:

[
100 + 100 + 25 = 225
]

MSE:

[
\frac{225}{3}=75
]

---

### Medium (Problem 6)

A decrease in MSE from **150** to **20** indicates that the model's predictions are, on average, much closer to the actual values. The model has improved significantly.

---

### Challenge

7. Summing raw errors is misleading because positive and negative errors can cancel each other, making a poor model appear perfect.

8. **MSE** emphasizes larger errors because of squaring, making it sensitive to outliers. **MAE** treats all errors proportionally and is often preferred when robustness to outliers is important.

---

# 20. Solution to Coding Assignment

```python
import numpy as np

def linear_regression_predict(X, w, b):
    return w * X + b

def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Dataset
X = np.array([800, 1000, 1200, 1500, 1800])
y = np.array([150, 180, 210, 260, 300])

# Try different parameters
models = [
    (0.15, 20),
    (0.16, 15),
    (0.17, 10)
]

for w, b in models:
    y_pred = linear_regression_predict(X, w, b)
    mse = compute_mse(y, y_pred)
    print(f"w={w:.2f}, b={b}, MSE={mse:.2f}")
```

Try adding more parameter combinations and identify the one with the smallest MSE.

---

# 21. Summary

Today you learned:

* ✅ What **Linear Regression** is and why it exists.
* ✅ The concepts of **features**, **targets**, **weights**, and **bias**.
* ✅ The **hypothesis function** (\hat{y}=wx+b).
* ✅ How to measure prediction quality using the **Cost Function**.
* ✅ Why **Mean Squared Error (MSE)** is commonly used.
* ✅ How to compute predictions and MSE from scratch in Python using NumPy.
* ✅ The intuition behind the **Normal Equation** and how it compares to **Gradient Descent**.

### What's Next?

**Day 4: Calculus Refresher**

We'll build the mathematical foundation needed for optimization by covering:

* Limits (brief intuition)
* Derivatives
* Partial Derivatives
* Gradients
* Chain Rule (preview)

These concepts will prepare you to fully understand **Gradient Descent** on Day 5, where you'll see how a machine automatically learns the best values of (w) and (b).

