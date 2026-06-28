

# Machine Learning From First Principles

# Week 1 — Day 1

# Chapter 1: What is Machine Learning?

> *"A computer is only as intelligent as the patterns it has learned."*

---

# Learning Objectives

By the end of this chapter, you will be able to:

* Explain what Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL) are.
* Understand why Machine Learning was invented.
* Distinguish between traditional programming and machine learning.
* Explain supervised and unsupervised learning.
* Differentiate regression and classification problems.
* Understand features (**X**) and targets (**y**).
* Describe the complete machine learning pipeline.
* Identify real-world applications of machine learning.

---

# Story-Based Introduction

## The Mango Farmer

Imagine a mango farmer who has harvested **50,000 mangoes**.

Every day, customers ask:

> "Is this mango sweet?"

The farmer knows from experience that sweetness depends on several factors:

* Weight
* Color
* Size
* Smell
* Ripeness

After years of experience, the farmer can almost instantly predict whether a mango will be sweet.

Now imagine hiring someone new.

You cannot transfer years of intuition directly.

Instead, you create a notebook.

| Weight | Color  | Size   | Sweet? |
| ------ | ------ | ------ | ------ |
| 220 g  | Yellow | Large  | Yes    |
| 180 g  | Green  | Medium | No     |
| 250 g  | Yellow | Large  | Yes    |

After reading thousands of examples, the new worker begins making accurate predictions.

Machine learning works in exactly the same way.

Instead of giving a computer explicit rules, we provide examples.

The computer learns patterns from those examples.

---

# Why Was Machine Learning Invented?

## Traditional Programming

For many problems, writing rules manually is straightforward.

For example:

```
If temperature > 100°C:
    Display "Boiling"
Else:
    Display "Not Boiling"
```

These rules are clear and deterministic.

However, consider identifying whether an email is spam.

Would you write rules like:

* Contains "Free"
* Contains "$$$"
* Too many links
* Too many capital letters
* Suspicious sender

Spammers quickly adapt.

The rules become outdated.

Maintaining them becomes nearly impossible.

---

## The Core Problem

Many real-world problems have **too many hidden patterns**.

Examples include:

* Face recognition
* Speech recognition
* Stock prediction
* Disease diagnosis
* Self-driving cars
* Language translation

Humans cannot manually write millions of rules for these tasks.

Instead, we let computers **learn** the rules from data.

---

# Intuition

Machine learning is like teaching by examples instead of instructions.

Consider teaching a child what a cat looks like.

You do not say:

* Two ears
* Four legs
* Tail exactly 25 cm
* Fur density 2000 hairs/cm²

Instead, you simply show many cats.

Eventually, the child recognizes cats naturally.

Machine learning follows the same principle.

Examples become the teacher.

---

# Visual Explanation

## Traditional Programming

```
Rules + Data
      │
      ▼
 Computer
      │
      ▼
 Output
```

---

## Machine Learning

```
Data + Correct Answers
          │
          ▼
 Learning Algorithm
          │
          ▼
      Model
          │
          ▼
 New Data
          │
          ▼
 Prediction
```

---

# Feynman Explanation

Imagine your younger sibling asks:

> "What is Machine Learning?"

You might say:

> Normally, we tell computers exactly what to do.

For example:

```
2 + 3 = 5
```

The computer follows instructions.

Machine learning is different.

Instead of giving instructions, we give examples.

The computer notices patterns.

Once it learns those patterns, it can make predictions about things it has never seen before.

Machine learning is **learning from experience instead of memorizing instructions**.

---

# Artificial Intelligence, Machine Learning, and Deep Learning

These terms are often confused.

```
Artificial Intelligence
│
├── Machine Learning
│      │
│      └── Deep Learning
```

---

## Artificial Intelligence (AI)

AI is the broad field of creating systems that perform tasks requiring human intelligence, such as reasoning, planning, understanding language, and making decisions.

Examples:

* Chess engines
* Voice assistants
* Route planning
* Robotics

AI is the umbrella.

---

## Machine Learning (ML)

Machine Learning is a subset of AI.

Instead of manually programming every rule, systems learn patterns from data.

Examples:

* Spam detection
* House price prediction
* Fraud detection
* Product recommendations

---

## Deep Learning (DL)

Deep Learning is a subset of Machine Learning.

It uses neural networks with many layers to automatically learn complex representations from data.

Examples:

* Chatbots
* Image generation
* Speech recognition
* Autonomous driving perception

---

# Mathematics Foundations

Machine learning often begins with a dataset.

We represent it mathematically as:

[
D = {(x_i, y_i)}_{i=1}^{n}
]

Where:

* (D) = dataset
* (n) = number of training examples
* (x_i) = input (features) for the (i^{th}) example
* (y_i) = correct output (target or label)

At this stage, think of this notation as a compact way to describe a collection of examples. We will gradually build the mathematical meaning throughout the course.

---

# Features (X) and Target (y)

Suppose we want to predict house prices.

| Size (sq ft) | Bedrooms | Age | Price       |
| ------------ | -------- | --- | ----------- |
| 1200         | 2        | 15  | ₹45 lakh    |
| 1800         | 3        | 5   | ₹72 lakh    |
| 2400         | 4        | 2   | ₹1.05 crore |

The inputs:

```
Size
Bedrooms
Age
```

are called **features**.

We usually denote them by:

[
X
]

The output:

```
Price
```

is called the **target**.

We denote it by:

[
y
]

So our learning problem is:

[
X \longrightarrow y
]

The model learns a function that maps features to the target.

---

# Types of Machine Learning

## 1. Supervised Learning

The computer learns using data where the correct answers are already known.

```
Input → Correct Answer
```

Examples:

* House price prediction
* Exam score prediction
* Spam detection
* Disease diagnosis

The teacher provides both the questions and the answers.

---

## 2. Unsupervised Learning

The computer receives only the inputs.

There are no correct answers.

Its task is to discover hidden patterns, groups, or structures in the data.

Examples:

* Customer segmentation
* News article clustering
* Anomaly detection

The computer acts like an explorer, organizing information without guidance.

---

# Regression vs Classification

## Regression

The output is a **continuous numerical value**.

Examples:

* Temperature
* House price
* Salary
* Fuel consumption

Example:

```
Predict house price

₹42,35,000
```

---

## Classification

The output belongs to one of several categories.

Examples:

* Spam / Not Spam
* Cancer / No Cancer
* Dog / Cat
* Fraud / Genuine

Example:

```
Email

Spam
```

---

# Machine Learning Pipeline

A typical machine learning project follows these steps:

```
Problem Definition
        │
        ▼
Collect Data
        │
        ▼
Clean Data
        │
        ▼
Feature Engineering
        │
        ▼
Train Model
        │
        ▼
Evaluate Model
        │
        ▼
Improve Model
        │
        ▼
Deploy
        │
        ▼
Monitor & Update
```

Each stage answers a different question:

* **Problem Definition:** What are we trying to predict?
* **Collect Data:** What information do we need?
* **Clean Data:** Is the data reliable?
* **Feature Engineering:** Which inputs help the model?
* **Train Model:** How can the model learn patterns?
* **Evaluate Model:** Does it perform well on unseen data?
* **Improve Model:** Can we reduce errors?
* **Deploy:** How do users access it?
* **Monitor & Update:** Does it remain accurate as the real world changes?

---

# Worked Examples

## Example 1: House Price Prediction

Input:

* Size
* Bedrooms
* Location

Output:

```
₹58 lakh
```

This is **supervised regression**.

---

## Example 2: Email Spam Detection

Input:

Email text

Output:

```
Spam
```

or

```
Not Spam
```

This is **supervised classification**.

---

## Example 3: Customer Segmentation

Input:

Customer purchase history

Output:

```
Group A
Group B
Group C
```

No labels are provided beforehand.

This is **unsupervised learning**.

---

# Python Implementation

We won't build a learning algorithm yet, but we can represent a small dataset using NumPy.

```python
import numpy as np

# Features: [House Size, Bedrooms]
X = np.array([
    [1200, 2],
    [1800, 3],
    [2400, 4]
])

# Target: House Price (in lakh rupees)
y = np.array([45, 72, 105])

print("Features:")
print(X)

print("\nTargets:")
print(y)
```

### Explanation

* `import numpy as np` imports the NumPy library, which will become our primary tool for numerical computing.
* `X` is a two-dimensional array where each row is one house and each column is a feature.
* `y` is a one-dimensional array containing the corresponding prices.
* We separate inputs (`X`) from outputs (`y`) because the model learns the relationship between them.

---

# Common Mistakes

1. Confusing AI with Machine Learning.
2. Assuming Deep Learning is different from Machine Learning rather than a subset of it.
3. Thinking Machine Learning "understands" data instead of learning statistical patterns.
4. Believing more data always guarantees a better model.
5. Mixing up features (`X`) and the target (`y`).
6. Using regression for categorical outputs or classification for continuous values.

---

# Practice Problems

## Easy

1. Define Machine Learning in your own words.
2. What is the difference between AI and ML?
3. Give three examples of supervised learning.
4. What are features?
5. What is the target variable?

## Medium

1. Explain why traditional programming struggles with spam detection.
2. Classify the following as regression or classification:

   * Predicting rainfall tomorrow.
   * Predicting whether a loan will default.
   * Predicting a student's exam score.
3. Describe the stages of a machine learning pipeline.

## Challenge

A hospital wants to:

* Predict the number of days a patient will stay.
* Detect whether an X-ray shows pneumonia.
* Group patients with similar symptoms when no diagnoses are available.

Identify the appropriate learning type (supervised or unsupervised) and, where applicable, whether the supervised task is regression or classification. Justify each choice.

---

# Coding Assignment

Create a small NumPy dataset containing information about at least **10 students**.

Include the following features:

* Hours studied
* Attendance percentage
* Practice tests completed

The target should be:

* Final exam score

Store the features in `X` and the scores in `y`. Print their shapes and explain what each dimension represents.

---

# Interview Questions

1. What is Machine Learning?
2. Why was Machine Learning invented?
3. Explain the difference between AI, Machine Learning, and Deep Learning.
4. What is the difference between supervised and unsupervised learning?
5. Explain regression and classification with examples.
6. What are features and targets?
7. Walk through the steps of a typical machine learning pipeline.
8. Why is data considered the foundation of machine learning?

---

# Reflection

Before moving to Day 2, ask yourself:

* Could I explain Machine Learning to someone with no technical background?
* Can I distinguish AI, ML, and Deep Learning without referring to notes?
* If someone gives me a real-world problem, can I decide whether it is supervised or unsupervised?
* Can I determine whether a supervised task is regression or classification?
* Do I understand why we represent data as features (`X`) and targets (`y`)?

If you can confidently answer "yes" to these questions, you have built the conceptual foundation for the rest of the course.

---

# Summary

In this chapter, you learned that machine learning is the science of enabling computers to learn patterns from data instead of relying solely on hand-written rules. You explored the relationship between AI, Machine Learning, and Deep Learning, learned the distinction between supervised and unsupervised learning, understood regression and classification, identified features (`X`) and targets (`y`), and saw how a complete machine learning pipeline transforms raw data into practical predictive systems.

This chapter establishes the mental model that every subsequent algorithm in the course will build upon. Before learning *how* algorithms work, you now understand *why* they exist and *what problems* they are designed to solve.
