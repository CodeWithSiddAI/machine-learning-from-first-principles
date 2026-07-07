# Week 2 – Day 8 (Part 1)

# Introduction to Classification

## Topic 1: Why Regression Fails for Classification?

---

# 1. Learning Objectives

By the end of this lesson, you will be able to:

* Understand what a **classification problem** is.
* Differentiate between **regression** and **classification**.
* Explain why **Linear Regression cannot solve classification problems effectively**.
* Recognize situations where classification is the appropriate ML approach.
* Develop the intuition that leads naturally to **Logistic Regression**, which we'll study in the next parts.

---

# 2. Story-Based Introduction

Imagine you are the owner of a hospital.

Every day, patients arrive with symptoms such as:

* Fever
* Blood pressure
* Oxygen level
* Heart rate
* Age

Your doctor asks you to build an AI system.

The system should answer only one question:

> **Does this patient have heart disease?**

Possible outputs:

* Yes
* No

Now imagine you use the Linear Regression model that we learned last week.

It predicts:

```
Patient A → 0.12
Patient B → 0.81
Patient C → 1.45
Patient D → -0.37
```

Immediately, several questions arise:

* What does **1.45 disease** mean?
* Can someone have **-0.37 disease**?
* Is **0.81** considered "Yes" or "No"?
* Why are values outside the range of valid answers?

Something clearly feels wrong.

The problem isn't with Linear Regression itself—it is being asked to solve the wrong kind of problem.

This realization led researchers to develop a new family of machine learning algorithms specifically for predicting **categories instead of numbers**.

That family is called **Classification**.

---

# 3. Why Was Classification Invented?

Before introducing any algorithm, let's answer the most important questions.

### What problem existed?

Many real-world tasks do **not** require predicting a continuous number.

Instead, they require choosing among a limited set of categories.

Examples:

| Problem           | Desired Output            |
| ----------------- | ------------------------- |
| Email filtering   | Spam / Not Spam           |
| Disease diagnosis | Sick / Healthy            |
| Loan approval     | Approve / Reject          |
| Face Unlock       | Authorized / Unauthorized |
| Credit card fraud | Fraud / Genuine           |
| Customer churn    | Leave / Stay              |

Notice something interesting.

None of these answers are measurements.

They are **labels**.

---

### What would happen if Classification didn't exist?

Suppose Gmail only had Linear Regression.

Instead of saying:

```
Spam
```

it might predict

```
2.87
```

What does that even mean?

Or a bank asks:

> Should we approve this loan?

Regression answers:

```
0.61
```

Approve?

Reject?

Nobody knows.

Regression is trying to solve a decision-making problem using continuous numbers.

This mismatch makes it unreliable.

---

### Where is Classification used?

Classification is everywhere.

* Medical diagnosis
* Face recognition
* Fingerprint unlocking
* Voice assistants
* Credit scoring
* Fraud detection
* Email spam filtering
* Product recommendation
* Image recognition
* Autonomous vehicles
* Customer segmentation (when labels are known)

If a computer must **choose among categories**, it is likely a classification problem.

---

# 4. Regression vs Classification

This is one of the most important distinctions in machine learning.

| Regression        | Classification      |
| ----------------- | ------------------- |
| Predicts numbers  | Predicts categories |
| Continuous output | Discrete output     |
| House prices      | Spam detection      |
| Temperature       | Disease diagnosis   |
| Salary            | Loan approval       |
| Stock price       | Cat vs Dog          |

### Visual Representation

```
Regression

House Size ---------------------> Price

200 sq ft ----------------------> ₹5,00,000
500 sq ft ----------------------> ₹9,00,000
700 sq ft ----------------------> ₹12,00,000


Outputs can be ANY real number.
```

---

Classification

```
Student Marks

35 -----> Fail

52 -----> Pass

90 -----> Pass

15 -----> Fail
```

Outputs are not numbers.

They are categories.

---

# 5. Intuition

Imagine a fruit seller.

Customers ask:

```
Is this fruit an Apple?
```

Possible answers:

```
Yes

No
```

Nobody answers

```
0.47 Apple
```

or

```
3.8 Apples
```

The answer belongs to one of two groups.

That is exactly what classification does.

It separates data into predefined classes.

---

Another example:

A security guard at an office entrance.

Every person arriving is either:

```
Employee

Visitor
```

The guard never says:

```
62% Employee
```

The final decision is categorical.

Machine learning models often need to make the same kind of decision.

---

# 6. Visual Explanation

Imagine plotting two types of data.

```
                Disease

        X X X X X X

---------------------------------

        O O O O O O

               Healthy
```

Where:

```
X = Disease

O = Healthy
```

The goal of a classification algorithm is to learn a rule that separates these two groups.

At this stage, don't worry about *how* that separation is made. We first need to understand *why* we need such a separator. In the next lesson, we'll introduce the idea of a **decision boundary**.

---

# 7. Feynman Explanation

Imagine explaining this to a 10-year-old.

Suppose I give you pictures of animals.

Some are cats.

Some are dogs.

I ask:

```
Tell me whether each picture is a cat or a dog.
```

There are only two possible answers.

Now imagine someone instead says:

```
This picture equals 7.8.
```

That number doesn't answer the question.

The task isn't to measure something.

The task is to **choose a label**.

Regression measures.

Classification chooses.

That's the entire difference.

---

# 8. Mathematical Foundations

## Regression Output

From Week 1, we learned that Linear Regression predicts:

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b
$$

Where:

* $\hat{y}$: predicted output
* $w_i$: learned weights for each feature
* $x_i$: input features
* $b$: bias (intercept)

Because the expression is a weighted sum, the output can be **any real number**:

* 250
* -13.7
* 0.001
* 1,000,000

This is perfect for regression problems, but problematic when valid answers are only categories such as **0 or 1**.

This mismatch is the motivation for Logistic Regression, which transforms this unrestricted output into a value between 0 and 1.

---

# 9. Worked Examples

### Example 1: House Price

Input:

```
Area = 1200 sq ft
Bedrooms = 3
```

Prediction:

```
₹72,50,000
```

Regression works perfectly because the output is continuous.

---

### Example 2: Spam Detection

Input:

```
Email:
"You won ₹10,00,000!"
```

Expected output:

```
Spam
```

Not:

```
2.91
```

This is a classification task.

---

### Example 3: Disease Prediction

Input:

```
Age = 55
BP = High
Heart Rate = 102
```

Expected output:

```
Heart Disease = Yes
```

Again, the output is a category rather than a number.

---

# 10. Python Implementation

Let's compare the types of outputs we expect.

```python
# Regression outputs
house_prices = [4500000, 6200000, 8100000]

print("Regression Predictions:")
for price in house_prices:
    print(price)
```

**Output**

```
Regression Predictions:
4500000
6200000
8100000
```

Now compare this with classification.

```python
# Classification outputs
predictions = ["Spam", "Not Spam", "Spam"]

print("Classification Predictions:")
for prediction in predictions:
    print(prediction)
```

**Output**

```
Classification Predictions:
Spam
Not Spam
Spam
```

### Explanation

* In regression, predictions are numerical values.
* In classification, predictions are labels representing categories.
* Later, Logistic Regression will internally compute probabilities and convert them into these labels.

---

# 11. Common Mistakes

1. **Thinking classification predicts numbers.**
   It predicts categories, even if those categories are encoded as 0 and 1.

2. **Using Linear Regression for classification.**
   Linear Regression can produce values outside the valid class range.

3. **Assuming binary classification is the only type.**
   Many classification problems involve three or more classes, which we'll cover later.

---

# 12. Practice Problems

## Easy

1. Is predicting a person's height regression or classification? Why?
2. Is spam detection regression or classification?
3. Is predicting tomorrow's temperature regression or classification?

## Medium

4. Explain why predicting a student's final exam score is different from predicting whether they pass or fail.
5. Give five real-world examples where classification is appropriate.

## Challenge

6. Design a simple classification problem from everyday life. Identify:

   * Inputs (features)
   * Possible output classes
   * Why regression would not be suitable

---

# 13. Coding Assignment

Write a Python program that:

1. Creates a list of 10 students with their exam marks.
2. Assigns the label **"Pass"** if the mark is at least 40; otherwise **"Fail"**.
3. Prints each student's mark along with the assigned label.

**Bonus:** Count how many students passed and how many failed.

---

# 14. Solutions for Practice Problems

1. Regression, because height is a continuous measurement.
2. Classification, because the output is either "Spam" or "Not Spam."
3. Regression, because temperature can take many continuous values.
4. Exam score is a numeric prediction (regression), while pass/fail is a categorical decision (classification).
5. Possible examples include disease diagnosis, loan approval, sentiment analysis, fraud detection, and image classification.
6. One example is predicting whether a customer will renew a subscription. Features might include usage time, support tickets, and subscription length. Outputs are "Renew" or "Not Renew," making it a classification task.

---

# 15. Solution for Coding Assignment

```python
marks = [25, 42, 78, 36, 90, 55, 18, 67, 49, 33]

passed = 0
failed = 0

for mark in marks:
    if mark >= 40:
        label = "Pass"
        passed += 1
    else:
        label = "Fail"
        failed += 1

    print(f"Mark: {mark:2d} -> {label}")

print("\nSummary")
print(f"Passed: {passed}")
print(f"Failed: {failed}")
```

**Expected Output**

```
Mark: 25 -> Fail
Mark: 42 -> Pass
Mark: 78 -> Pass
...
Summary
Passed: 6
Failed: 4
```

---

# 16. Summary

Today, we established the motivation for classification by identifying the limitations of regression on categorical tasks.

Key takeaways:

* **Regression** predicts continuous numerical values.
* **Classification** predicts discrete categories or labels.
* Many real-world problems—such as spam detection, medical diagnosis, fraud detection, and loan approval—are naturally classification problems.
* Linear Regression is unsuitable for these tasks because it can produce unrestricted outputs, including impossible values for categorical decisions.
* This limitation motivates the need for **Logistic Regression**, which we will begin exploring next.

---

## Looking Ahead (Part 2)

In the next lesson, we'll answer an important question:

> **How does a machine learning model actually separate one class from another?**

This will introduce the concept of the **Decision Boundary**, the geometric idea that lies at the heart of almost every classification algorithm.
