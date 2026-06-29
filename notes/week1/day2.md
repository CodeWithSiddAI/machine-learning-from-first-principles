Excellent. Welcome to **Week 1 – Day 2**.

This is one of the **most important days in the entire course**.

Many beginners think Machine Learning is about algorithms. It isn't.

Machine Learning is **linear algebra implemented in code**.

If you become comfortable with today's topics, then Linear Regression, Neural Networks, Transformers, and even Large Language Models will become much easier to understand.

---

# Week 1 – Day 2 Roadmap

Today's lesson follows the same structure as Day 1:

1. **Learning Objectives**
2. **Story-based Introduction**
3. **Intuition**
4. **Visual Explanation**
5. **Feynman Explanation**
6. **Mathematical Foundations**
7. **Worked Examples**
8. **Python Implementation (NumPy)**
9. **Common Mistakes**
10. **Practice Problems**
11. **Coding Assignment**
12. **Interview Questions**
13. **Reflection**
14. **Summary**

---

# Topics Covered Today

### Part 1 — Python for Machine Learning

* Variables
* Lists vs NumPy arrays
* Functions
* Loops
* Why Python became the language of ML

---

### Part 2 — NumPy

* Creating arrays
* Shape
* Dimensions
* Indexing
* Slicing
* Useful operations

---

### Part 3 — Vectors

* What is a vector?
* Row vector
* Column vector
* Vector addition
* Scalar multiplication
* Magnitude (intuition only)

---

### Part 4 — Matrices

* Matrix notation
* Rows
* Columns
* Matrix shapes
* Matrix operations

---

### Part 5 — Dot Product

The single most important operation in Machine Learning.

You'll learn:

* Why it exists
* Intuition
* Formula
* Geometric meaning
* ML interpretation
* NumPy implementation

---

### Part 6 — Broadcasting

One of NumPy's "magic" features.

You'll understand:

* Why broadcasting was invented
* Broadcasting rules
* Examples
* Common mistakes

---

# Expected Outcome

By the end of Day 2, you'll be able to:

* Read almost every NumPy-based ML implementation.
* Understand how datasets are stored in memory.
* Explain vectors and matrices intuitively.
* Compute dot products by hand and with NumPy.
* Use broadcasting without confusion.
* Be fully prepared for implementing **Linear Regression from scratch on Day 3**.

---

## A Small Preview

Tomorrow (Day 3), we'll write equations like:

[
\hat{y} = Xw + b
]

Right now, this may look mysterious.

By the end of Day 2, every symbol in that equation will make sense:

* (X): the feature matrix
* (w): the vector of model weights
* (b): the bias term
* (Xw): a dot product between the data and the weights
* (\hat{y}): the model's predicted output

This equation is the foundation of linear regression and, with extensions, much of modern machine learning.

---

We'll take the same intuition-first approach as Day 1, building each concept from simple ideas to mathematics to code. By the end of today, you'll have the mathematical language that nearly every machine learning algorithm speaks.


Great! Let's begin **Week 1 – Day 2**.

---

# Machine Learning From First Principles

# Week 1 – Day 2

# Chapter 2: Python & NumPy for Machine Learning (Part 1)

> *"Before you teach a machine to learn, you must first learn how to speak its language."*

---

# Learning Objectives

By the end of this lesson, you will be able to:

* Understand why Python is the most popular language for Machine Learning.
* Write simple Python programs confidently.
* Understand variables, data types, and operators.
* Use lists and understand their limitations.
* Write conditional statements and loops.
* Create and use functions.
* Understand why we need NumPy.

---

# Story-Based Introduction

## The Chef and the Kitchen

Imagine you're training to become a professional chef.

Before you cook a five-course meal, you must first learn:

* How to hold a knife
* How to use a stove
* How to measure ingredients
* How to use kitchen tools

These skills don't make you a chef by themselves—but without them, cooking becomes difficult.

Machine Learning is no different.

Before we build intelligent models, we need the right tools.

Python is our **kitchen**.

NumPy is our **knife**.

Machine Learning algorithms are the **recipes** we'll learn later.

If your tools are slow or awkward, even a simple recipe becomes frustrating. That's why we start by mastering the essentials.

---

# Why Python?

## Why wasn't Machine Learning built around C or Java?

Python became the dominant language because it is:

* Easy to read
* Easy to write
* Supported by a huge ecosystem
* Great for mathematics
* Backed by thousands of ML libraries
* Used by researchers and industry alike

For example, compare these two snippets.

### Python

```python
x = 10
y = 20
print(x + y)
```

### Java (conceptually)

```java
int x = 10;
int y = 20;
System.out.println(x + y);
```

Python lets us focus on solving problems instead of writing boilerplate code.

---

# Intuition

Think of Python as **English for computers**.

Humans communicate with words.

Computers communicate with instructions.

Python lets us express those instructions in a way that is easy for humans to read and write.

Instead of thinking about complex computer hardware, we describe *what* we want to do, and Python handles much of the complexity.

---

# Visual Explanation

```text
You
 │
 ▼
Python Code
 │
 ▼
Python Interpreter
 │
 ▼
Computer
 │
 ▼
Result
```

You write Python.

The interpreter translates it into instructions the computer understands.

---

# Feynman Explanation

Imagine explaining Python to a 10-year-old.

You might say:

> Python is a language that lets us tell computers what to do using simple, readable sentences.

Instead of flipping millions of tiny electrical switches ourselves, we write instructions like:

```python
print("Hello")
```

The computer reads those instructions and performs the task.

Python is simply a convenient way for humans and computers to communicate.

---

# Variables

## Why do we need variables?

Suppose you're calculating a student's marks.

Without variables:

```python
85 + 10
```

What does 85 mean?

Math?

Science?

English?

It's unclear.

Variables give names to values.

```python
marks = 85
bonus = 10

total = marks + bonus
```

Now the code tells a story.

---

# Mathematics Behind Variables

Think of a variable as a labeled box.

```text
+---------+
| marks   |
|   85    |
+---------+
```

The label is the variable name.

The value inside the box can change.

---

# Data Types

Python stores different kinds of information.

| Type    | Example              |
| ------- | -------------------- |
| Integer | `10`                 |
| Float   | `3.14`               |
| String  | `"Machine Learning"` |
| Boolean | `True`               |

Examples:

```python
age = 25
height = 1.75
name = "Alice"
passed = True
```

---

# Operators

Python supports mathematical operations.

```python
x = 10
y = 3

print(x + y)   # 13
print(x - y)   # 7
print(x * y)   # 30
print(x / y)   # 3.333...
print(x // y)  # 3
print(x % y)   # 1
print(x ** y)  # 1000
```

---

# Lists

Suppose we have marks for five students.

Without lists:

```python
student1 = 82
student2 = 75
student3 = 91
student4 = 64
student5 = 88
```

This quickly becomes difficult to manage.

Instead:

```python
marks = [82, 75, 91, 64, 88]
```

Now all related values are stored together.

---

# Accessing List Elements

Python starts counting from **0**, not 1.

```python
marks = [82, 75, 91, 64, 88]

print(marks[0])   # 82
print(marks[1])   # 75
print(marks[4])   # 88
```

Visualization:

```text
Index:   0    1    2    3    4
        -------------------------
Marks = [82 | 75 | 91 | 64 | 88]
```

This is called **zero-based indexing**.

---

# Why Lists Are Not Enough

Lists are useful, but Machine Learning often involves **millions of numbers**.

Imagine storing:

* 1 million house prices
* 1 million images
* 1 million medical records

Python lists become:

* Slower
* Less memory efficient
* Harder to perform mathematical operations on

That's why we use **NumPy**, which we'll introduce later in today's lesson.

---

# Conditional Statements

Sometimes a program needs to make decisions.

Example:

```python
marks = 72

if marks >= 50:
    print("Pass")
else:
    print("Fail")
```

Think of this as asking a yes/no question.

If the condition is true, one block of code runs.

Otherwise, another block runs.

---

# Loops

Imagine printing numbers from 1 to 100.

Without a loop:

```python
print(1)
print(2)
print(3)
...
```

This is repetitive.

A loop automates repetition.

```python
for i in range(1, 6):
    print(i)
```

Output:

```text
1
2
3
4
5
```

---

# Functions

Suppose you need to calculate the area of many rectangles.

Instead of rewriting the formula each time:

```python
length = 5
width = 3
area = length * width

length = 8
width = 2
area = length * width
```

Create a function:

```python
def rectangle_area(length, width):
    return length * width
```

Now use it whenever needed:

```python
print(rectangle_area(5, 3))
print(rectangle_area(8, 2))
```

Functions help us:

* Reuse code
* Avoid repetition
* Improve readability
* Reduce errors

---

# Worked Examples

### Example 1

```python
temperature = 35

if temperature > 30:
    print("Hot")
else:
    print("Cool")
```

**Explanation:**

* The variable `temperature` stores the value `35`.
* The `if` statement checks whether `temperature > 30`.
* Since `35 > 30`, the condition is **True**.
* The program prints:

```text
Hot
```

---

### Example 2

```python
numbers = [10, 20, 30]

for number in numbers:
    print(number)
```

Output:

```text
10
20
30
```

The loop visits each element of the list one by one and prints it.

---

### Example 3

```python
def square(x):
    return x * x

print(square(6))
```

Output:

```text
36
```

The function receives `6`, multiplies it by itself, and returns `36`.

---

# Common Mistakes

1. Forgetting that Python uses **0-based indexing**.
2. Mixing tabs and spaces for indentation.
3. Using `=` (assignment) when you mean `==` (comparison).
4. Forgetting the colon (`:`) after `if`, `for`, or `def`.
5. Calling a function without the required arguments.

---

# Practice Problems

## Easy

1. Create variables for your name, age, and favorite number, then print them.
2. Create a list of five fruits and print the first and last fruit.
3. Write an `if` statement that prints `"Adult"` if age is 18 or more; otherwise print `"Minor"`.

---

## Medium

1. Write a `for` loop that prints the numbers from 1 to 10.
2. Write a function `cube(x)` that returns the cube of a number.
3. Create a list of five numbers and calculate their sum using a loop.

---

## Challenge

Write a program that:

* Stores the marks of five students in a list.
* Calculates the average mark.
* Prints `"Excellent"` if the average is at least 90.
* Prints `"Good"` if the average is between 70 and 89.
* Otherwise prints `"Needs Improvement"`.

---

# Coding Assignment

Create a simple **Student Report Card** program.

Requirements:

* Store a student's name and marks in three subjects.
* Calculate the total and average.
* Print whether the student passed (average ≥ 50).
* Organize your code by writing a function that computes the average.

---

# Interview Questions

1. Why is Python the preferred language for Machine Learning?
2. What is a variable?
3. What is the difference between a list and individual variables?
4. Why does Python use zero-based indexing?
5. What is the purpose of a function?
6. When would you use a `for` loop instead of writing repeated code?

---

# Reflection

Before moving to NumPy, ask yourself:

* Can I explain why Python is widely used in Machine Learning?
* Do I understand variables, data types, and lists?
* Can I write simple `if` statements and `for` loops?
* Do I know why functions are useful?
* Can I read and understand basic Python code without guessing?

If the answer is **yes**, you're ready for the next part of Day 2: **NumPy**, where we'll learn to work with data efficiently and prepare for the mathematical foundations of machine learning.


Excellent. Since you're comfortable with basic Python, we'll move to the **heart of Day 2**.

---

# Machine Learning From First Principles

# Week 1 – Day 2

# Chapter 2 (Part 2): NumPy — The Language of Machine Learning

> *"If Python is the language of Machine Learning, NumPy is its mathematics."*

---

# Learning Objectives

By the end of this chapter, you will be able to:

* Explain why NumPy exists.
* Create NumPy arrays.
* Understand dimensions and shapes.
* Index and slice arrays.
* Perform basic mathematical operations.
* Explain why NumPy is much faster than Python lists.

---

# Story-Based Introduction

## The Library Assistant

Imagine you are the librarian of the world's largest library.

You need to organize:

* 10 books
* 1,000 books
* 10 million books

With only a notebook, keeping track of millions of books would be nearly impossible.

Instead, you use a computer database designed specifically for organizing large amounts of information efficiently.

Python lists are like the notebook.

NumPy arrays are like the professional database.

Machine Learning works with **millions or even billions of numbers**, so we need a tool designed for speed and efficiency.

That tool is **NumPy**.

---

# Why Was NumPy Invented?

Suppose you have one million student marks.

Python lists can store them:

```python
marks = [72, 85, 91, ...]
```

But what if you want to add 5 bonus marks to every student?

With a list, you'd need a loop:

```python
new_marks = []

for mark in marks:
    new_marks.append(mark + 5)
```

This works, but it's slow for very large datasets.

NumPy was created to perform operations on entire collections of numbers at once, making them much faster and more memory-efficient.

---

# Intuition

Think of a Python list as a row of separate boxes.

```text
[72] [85] [91] [66] [80]
```

Each box is handled individually.

A NumPy array is like a single block of connected memory.

```text
+--------------------------------+
|72|85|91|66|80|
+--------------------------------+
```

Because the numbers are stored together, the computer can process many of them simultaneously.

---

# Visual Explanation

```text
Python List

72   85   91   66   80
□    □    □    □    □

Stored as separate objects
```

```text
NumPy Array

+---------------------------+
|72|85|91|66|80|
+---------------------------+

Stored in one continuous block
```

This layout is one reason NumPy is so efficient.

---

# Feynman Explanation

Imagine explaining NumPy to a friend.

You could say:

> A Python list is like writing numbers on separate sticky notes.

> A NumPy array is like printing those numbers neatly in a spreadsheet.

Because everything is organized consistently, mathematical operations become much faster and easier.

---

# Installing NumPy

If you haven't already installed it:

```bash
pip install numpy
```

Import it into your program:

```python
import numpy as np
```

`np` is simply a short nickname for NumPy.

---

# Creating Arrays

Create a one-dimensional array:

```python
import numpy as np

numbers = np.array([10, 20, 30, 40])

print(numbers)
```

Output:

```text
[10 20 30 40]
```

Unlike a Python list, this is a **NumPy array**.

---

# Dimensions

A **dimension** tells us how many directions data extends.

### 1D Array

```python
numbers = np.array([10,20,30,40])
```

Visualization:

```text
10 20 30 40
```

One row.

One dimension.

---

### 2D Array

```python
matrix = np.array([
    [1,2,3],
    [4,5,6]
])
```

Visualization:

```text
1 2 3
4 5 6
```

Two dimensions:

* Rows
* Columns

---

### 3D Array

Think of multiple matrices stacked together.

```text
Matrix 1

1 2
3 4

Matrix 2

5 6
7 8
```

We'll use 3D arrays later for images and deep learning.

---

# Shape

Every NumPy array has a **shape**.

Example:

```python
matrix = np.array([
    [1,2,3],
    [4,5,6]
])

print(matrix.shape)
```

Output:

```text
(2, 3)
```

Meaning:

* 2 rows
* 3 columns

---

## More Examples

```python
np.array([1,2,3])
```

Shape:

```text
(3,)
```

Three elements.

One dimension.

---

```python
np.array([
 [1,2],
 [3,4],
 [5,6]
])
```

Shape:

```text
(3,2)
```

Three rows.

Two columns.

---

# Indexing

Exactly like Python lists.

```python
numbers = np.array([10,20,30,40])

print(numbers[0])
```

Output

```text
10
```

---

2D indexing:

```python
matrix = np.array([
 [1,2,3],
 [4,5,6]
])

print(matrix[1,2])
```

Output

```text
6
```

Remember:

```text
matrix[row, column]
```

---

# Visualization

```text
        Column

        0   1   2

Row 0   1   2   3

Row 1   4   5   6
```

So:

```python
matrix[1,2]
```

means:

Row 1

Column 2

Result:

```text
6
```

---

# Slicing

Suppose we have:

```python
numbers = np.array([10,20,30,40,50])
```

Get the first three numbers:

```python
numbers[:3]
```

Output

```text
[10 20 30]
```

---

Get the last two:

```python
numbers[-2:]
```

Output

```text
[40 50]
```

---

2D slicing

```python
matrix[:,1]
```

Visualization:

```text
1 2 3
4 5 6
```

This means:

* `:` → all rows
* `1` → second column

Output

```text
[2 5]
```

---

# Mathematical Operations

Here's where NumPy shines.

```python
numbers = np.array([10,20,30])

print(numbers + 5)
```

Output

```text
[15 25 35]
```

No loop required!

---

Multiply everything by 2:

```python
numbers * 2
```

Output

```text
[20 40 60]
```

---

Square everything:

```python
numbers ** 2
```

Output

```text
[100 400 900]
```

---

Add two arrays:

```python
a = np.array([1,2,3])
b = np.array([4,5,6])

print(a + b)
```

Output

```text
[5 7 9]
```

NumPy performs the addition element by element.

---

# Worked Example

Suppose student marks are:

```python
marks = np.array([70,80,90])
```

The teacher gives everyone 5 bonus marks.

```python
updated = marks + 5

print(updated)
```

Output

```text
[75 85 95]
```

No loop.

One line.

Very fast.

---

# Why NumPy Is Faster

Python list:

```text
One item
↓

Add

↓

Next item

↓

Add

↓

Next item
```

NumPy:

```text
Entire Array

↓

Single Optimized Operation

↓

Finished
```

Internally, NumPy is implemented using highly optimized C code and performs operations on contiguous blocks of memory, making it much faster than looping through Python objects one by one.

---

# Common Mistakes

1. Forgetting to import NumPy:

```python
import numpy as np
```

2. Confusing list indexing with matrix indexing.

3. Misreading the shape `(rows, columns)`.

4. Forgetting that indexing starts at `0`.

5. Trying to add arrays of incompatible shapes (we'll learn about this with broadcasting).

---

# Practice Problems

## Easy

1. Create a NumPy array containing the numbers `5, 10, 15, 20`.
2. Print its shape.
3. Print the second element.

---

## Medium

1. Create a 3×3 matrix and print the element in the second row, third column.
2. Create an array `[2, 4, 6, 8]` and multiply every element by `3`.
3. Slice the last three elements from the array `[10, 20, 30, 40, 50, 60]`.

---

## Challenge

A teacher has marks:

```text
65 70 75 80 85
```

Using NumPy:

* Add 10 bonus marks.
* Double the updated marks.
* Print the shape of the final array.
* Print only the first three values.

---

# Coding Assignment

Create a NumPy array representing the heights (in cm) of 10 students.

Then:

1. Print the array.
2. Print its shape.
3. Add 2 cm to every height.
4. Print the updated array.
5. Print the tallest height using NumPy.

---

# Interview Questions

1. Why is NumPy preferred over Python lists for Machine Learning?
2. What is a NumPy array?
3. What does the shape `(4, 3)` mean?
4. What is the difference between indexing and slicing?
5. Why can NumPy perform operations like `array + 5` without a loop?

---

# Reflection

Before continuing to **Vectors**, ask yourself:

* Can I explain why NumPy was invented?
* Do I understand the difference between a Python list and a NumPy array?
* Can I create 1D and 2D arrays?
* Can I find the shape of an array?
* Can I index and slice arrays confidently?
* Can I perform basic arithmetic on arrays without writing loops?

If you can answer "yes" to these questions, you're ready to move on to **Vectors**, where we'll begin the linear algebra that underpins every machine learning algorithm.


Excellent. Now we begin one of the **most important topics in all of Machine Learning**.

Almost every ML algorithm—from **Linear Regression** to **Neural Networks**, **Transformers**, and even **LLMs like ChatGPT**—represents data as **vectors**.

---

# Machine Learning From First Principles

# Week 1 – Day 2

# Chapter 2 (Part 3): Vectors

> *"A vector is not just a list of numbers—it is one complete description of a single object."*

---

# Learning Objectives

By the end of this chapter, you will be able to:

* Understand what a vector is.
* Explain why vectors are used in Machine Learning.
* Distinguish between scalars and vectors.
* Understand row vectors and column vectors.
* Perform vector addition and scalar multiplication.
* Develop an intuition for vector magnitude.

---

# Story-Based Introduction

## Describing a Student

Imagine your school asks you to describe a student.

You could write:

* Name: Alice
* Age: 20
* Height: 170 cm
* Weight: 60 kg
* Exam Score: 92

This works for one student.

But what if you have **10,000 students**?

Writing descriptions in sentences becomes difficult to process.

Instead, we can describe each student using only numbers:

```text
[20, 170, 60, 92]
```

This single collection of numbers completely describes one student (for the features we've chosen).

In Machine Learning, this collection is called a **vector**.

---

# Why Were Vectors Invented?

Imagine trying to predict house prices.

Each house has:

* Size
* Bedrooms
* Bathrooms
* Age
* Distance from city

Instead of storing each value separately:

```text
Size = 1800
Bedrooms = 3
Bathrooms = 2
Age = 8
Distance = 5 km
```

We group them together:

```text
[1800, 3, 2, 8, 5]
```

Now the computer treats this as **one object**.

This makes mathematical operations much simpler and more efficient.

---

# Intuition

Think of a vector as a **profile card**.

A person's passport summarizes their identity.

Similarly, a vector summarizes an object using numbers.

For example:

```text
Student A

Hours Studied
Attendance
Assignments

↓

[8, 95, 12]
```

This vector represents one student.

Another student:

```text
[4, 70, 6]
```

Each vector is one complete description.

---

# Visual Explanation

Imagine a spreadsheet.

| Student | Hours | Attendance | Assignments |
| ------- | ----- | ---------- | ----------- |
| Alice   | 8     | 95         | 12          |
| Bob     | 4     | 70         | 6           |
| Charlie | 10    | 98         | 15          |

Each **row** is a vector.

```text
Alice

[8 95 12]
```

Bob

```text
[4 70 6]
```

Charlie

```text
[10 98 15]
```

Machine Learning often represents each data sample as one vector.

---

# Feynman Explanation

Imagine explaining vectors to a child.

You could say:

> Suppose I ask you to describe your favorite cricket player.

You might say:

* Height
* Runs
* Age
* Jersey Number

Instead of saying all those words every time, you could simply write:

```text
[183, 12000, 36, 18]
```

That's a vector.

It's simply one object described using several numbers.

---

# Scalars vs Vectors

A **scalar** is a single number.

Examples:

```text
5

42

3.14
```

A **vector** is a collection of related numbers.

```text
[5, 8, 2]

[170, 60, 21]

[1800, 3, 2]
```

| Scalar      | Vector                  |
| ----------- | ----------------------- |
| One value   | Multiple related values |
| Temperature | Student record          |
| Age         | House features          |
| Price       | Image pixels            |

---

# Mathematical Foundations

A vector is commonly written in bold:

[
\mathbf{x}
==========

\begin{bmatrix}
2 \
5 \
8
\end{bmatrix}
]

Here:

* (\mathbf{x}) is the vector's name.
* Each entry is one feature.
* The vector has **3 components**, so it is a **3-dimensional vector**.

In machine learning, the dimension of a vector equals the number of features describing an example.

---

# Row Vector vs Column Vector

## Row Vector

[
\mathbf{x}
==========

\begin{bmatrix}
2 & 5 & 8
\end{bmatrix}
]

Shape:

```text
(1,3)
```

One row.

Three columns.

---

## Column Vector

[
\mathbf{x}
==========

\begin{bmatrix}
2 \
5 \
8
\end{bmatrix}
]

Shape:

```text
(3,1)
```

Three rows.

One column.

---

## Which One Do We Use?

In machine learning:

* **A single data sample** is often shown as a **row vector** in a dataset.
* **Weight vectors** in mathematical derivations are often written as **column vectors**.

You'll see both throughout the course, and later you'll learn why matrix multiplication often prefers one orientation over the other.

---

# Vector Addition

Suppose two students improve their scores in different subjects.

Original scores:

[
\mathbf{a} =
\begin{bmatrix}
70 \
80 \
90
\end{bmatrix}
]

Improvement:

[
\mathbf{b} =
\begin{bmatrix}
5 \
3 \
2
\end{bmatrix}
]

Add them component by component:

[
\mathbf{a} + \mathbf{b}
=======================

\begin{bmatrix}
75 \
83 \
92
\end{bmatrix}
]

### Rule

Vectors can only be added if they have the **same number of components**.

---

# Scalar Multiplication

Suppose every student's marks are doubled.

[
2
\begin{bmatrix}
70\
80\
90
\end{bmatrix}
=============

\begin{bmatrix}
140\
160\
180
\end{bmatrix}
]

Every component is multiplied by the scalar.

---

# Vector Magnitude (Intuition)

Imagine two arrows.

```text
Small Arrow

---->

Large Arrow

-------------->
```

The direction is the same.

Only the length changes.

That length is called the **magnitude** (or **norm**) of the vector.

In machine learning, magnitude often represents the "size" or "strength" of a vector.

We'll compute it formally when we study distance and similarity.

---

# Worked Examples

## Example 1

Student A:

```text
[80, 90, 70]
```

Student B:

```text
[10, 5, 20]
```

Add them:

```text
[90, 95, 90]
```

Each position is added independently.

---

## Example 2

Multiply:

```text
[3, 6, 9]
```

by

```text
4
```

Result:

```text
[12, 24, 36]
```

---

# Python Implementation

Create a vector:

```python
import numpy as np

student = np.array([80, 90, 70])

print(student)
```

Output:

```text
[80 90 70]
```

---

## Vector Addition

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(a + b)
```

Output:

```text
[5 7 9]
```

NumPy adds each corresponding element.

---

## Scalar Multiplication

```python
marks = np.array([70, 80, 90])

print(marks * 2)
```

Output:

```text
[140 160 180]
```

Each element is multiplied by 2.

---

## Magnitude (Preview)

```python
vector = np.array([3, 4])

print(np.linalg.norm(vector))
```

Output:

```text
5.0
```

This uses the famous 3–4–5 right triangle.

Don't worry about the formula yet—we'll revisit it later when discussing distances.

---

# Common Mistakes

1. Confusing a vector with a Python list.
2. Trying to add vectors of different lengths.
3. Mixing row vectors and column vectors.
4. Thinking a vector must represent a physical arrow. In machine learning, vectors usually represent **data**.

---

# Practice Problems

## Easy

1. Is `7` a scalar or a vector?
2. Is `[10, 20, 30]` a scalar or a vector?
3. How many components does the vector `[5, 9, 2, 8]` have?

---

## Medium

1. Add the vectors:

```text
[2, 4, 6]

+

[1, 3, 5]
```

2. Multiply:

```text
[3, 7, 9]
```

by

```text
5
```

3. Is it valid to add:

```text
[1, 2]

+

[3, 4, 5]
```

Why or why not?

---

## Challenge

A student is described by the vector:

```text
[Hours Studied, Attendance, Assignments]

[7, 92, 14]
```

Answer the following:

1. What does each component represent?
2. How many features describe the student?
3. If the student completes 2 more assignments, what is the new vector?
4. If all values are doubled, what does the resulting vector become?

---

# Coding Assignment

Using NumPy:

1. Create vectors for the marks of two students in three subjects.
2. Add the vectors.
3. Multiply one vector by `2`.
4. Print all results.
5. Print the shape of each vector.

---

# Interview Questions

1. What is a vector?
2. How is a vector different from a scalar?
3. Why are vectors important in machine learning?
4. What is the difference between a row vector and a column vector?
5. When can two vectors be added?
6. What does scalar multiplication do to a vector?

---

# Reflection

Before moving to **Matrices**, ask yourself:

* Can I explain a vector without mentioning formulas?
* Can I identify vectors in a dataset?
* Do I understand why each row of a dataset can be viewed as a vector?
* Can I perform vector addition and scalar multiplication by hand?
* Do I understand the difference between a scalar and a vector?

If you can answer **yes** to these questions, you're ready for **Matrices**—where we'll combine many vectors into a single structure and see how datasets are represented in machine learning.

---

### A Preview of What's Next

A **vector describes one object**.

A **matrix stores many vectors together**.

For example:

```text
Student 1 → [8, 95, 12]
Student 2 → [6, 88, 10]
Student 3 → [9, 97, 15]
```

Together, they form:

```text
[
 [8, 95, 12],
 [6, 88, 10],
 [9, 97, 15]
]
```

That structure—a collection of vectors—is called a **matrix**, and it is how almost every machine learning dataset is represented internally. This is our next step.
