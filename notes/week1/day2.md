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

$$
\mathbf{x} = 
\begin{bmatrix}
2 \\
5 \\
8
\end{bmatrix}
$$

Here:

* (\mathbf{x}) is the vector's name.
* Each entry is one feature.
* The vector has **3 components**, so it is a **3-dimensional vector**.

In machine learning, the dimension of a vector equals the number of features describing an example.

---

# Row Vector vs Column Vector

## Row Vector

$$
\mathbf{x}
=

\begin{bmatrix}
2 & 5 & 8
\end{bmatrix}
$$

Shape:

```text
(1,3)
```

One row.

Three columns.

---

## Column Vector

$$
\mathbf{x} = 
\begin{bmatrix}
2 \\
5 \\
8
\end{bmatrix}
$$

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

$$
\mathbf{a} =
\begin{bmatrix}
70 \\
80 \\
90
\end{bmatrix}
$$

Improvement:

$$
\mathbf{b} =
\begin{bmatrix}
5 \\
3 \\
2
\end{bmatrix}
$$

Add them component by component:

$$
\mathbf{a} + \mathbf{b}
=

\begin{bmatrix}
75 \\
83 \\
92
\end{bmatrix}
$$

### Rule

Vectors can only be added if they have the **same number of components**.

---

# Scalar Multiplication

Suppose every student's marks are doubled.

$$
2
\begin{bmatrix}
70\\
80\\
90
\end{bmatrix}
=

\begin{bmatrix}
140\
160\
180
\end{bmatrix}
$$

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


Excellent. Now we reach the section that separates someone who can **use** machine learning libraries from someone who **understands** them.

The next topic is arguably **the single most important concept in linear algebra for Machine Learning**.

---

# Machine Learning From First Principles

# Week 1 – Day 2

# Chapter 2 (Part 4): Matrices

> *"A vector describes one object. A matrix describes an entire dataset."*

---

# Learning Objectives

By the end of this chapter, you will be able to:

* Explain what a matrix is.
* Understand why matrices are fundamental to Machine Learning.
* Read and write matrix notation.
* Identify rows, columns, and matrix dimensions.
* Understand how datasets are represented as matrices.
* Perform basic matrix operations using NumPy.

---

# Story-Based Introduction

## The School Principal

Yesterday, we described **one student** using a vector.

For example:

$$
\mathbf{x} =
\begin{bmatrix}
8 \
95 \
12
\end{bmatrix}
$$

Where:

* 8 = Hours Studied
* 95 = Attendance (%)
* 12 = Assignments Completed

Now imagine the principal wants information for **every student** in the school.

Writing hundreds of vectors separately would be messy.

Instead, we stack them together.

$$
X =
\begin{bmatrix}
8 & 95 & 12 \
6 & 88 & 10 \
9 & 97 & 15 \
7 & 91 & 13
\end{bmatrix}
$$

Instead of describing **one student**, we now describe the **entire class**.

That collection is called a **matrix**.

---

# Why Were Matrices Invented?

Suppose you have data for **1 million houses**.

Each house has:

* Size
* Bedrooms
* Bathrooms
* Age
* Distance to City

Instead of storing each feature separately:

```python
sizes = [...]
bedrooms = [...]
bathrooms = [...]
ages = [...]
distances = [...]
```

We organize everything into one structure:

$$
X =
\begin{bmatrix}
1800 & 3 & 2 & 8 & 5 \
2200 & 4 & 3 & 2 & 7 \
1500 & 2 & 2 & 15 & 3
\end{bmatrix}
$$

Each row is one house.

Each column is one feature.

Now every machine learning algorithm can process the entire dataset efficiently.

---

# Intuition

Think of a spreadsheet.

Imagine opening Microsoft Excel.

| Student | Hours | Attendance | Assignments |
| ------- | ----: | ---------: | ----------: |
| Alice   |     8 |         95 |          12 |
| Bob     |     6 |         88 |          10 |
| Charlie |     9 |         97 |          15 |

That spreadsheet is essentially a **matrix**.

Machine learning algorithms don't see names like "Hours" or "Attendance."

They see numbers arranged in rows and columns.

---

# Visual Explanation

```text
             Features
          Hrs  Att  Assignments
          ----------------------
Student 1   8   95      12
Student 2   6   88      10
Student 3   9   97      15
Student 4   7   91      13
```

Notice two important ideas:

* **Rows represent training examples.**
* **Columns represent features.**

This convention will remain true throughout almost the entire course.

---

# Feynman Explanation

Imagine explaining matrices to a 12-year-old.

You could say:

> Yesterday, we learned that a vector is like one student's report card.

A matrix is simply a pile of report cards placed together in a table.

Instead of remembering one student,

we remember **everyone**.

Nothing magical happened.

We simply stacked vectors together.

---

# Mathematical Foundations

A matrix is usually represented by an uppercase bold letter.

For example:

$$
X =
\begin{bmatrix}
1 & 2 & 3 \
4 & 5 & 6
\end{bmatrix}
$$

This matrix has:

* **2 rows**
* **3 columns**

We say its **shape** is:

$$
(2,,3)
$$

---

## Matrix Dimensions

Suppose we have:

$$
A =
\begin{bmatrix}
3 & 7 \
5 & 9 \
2 & 8
\end{bmatrix}
$$

Count the rows:

* 3

Count the columns:

* 2

Therefore,

$$
A \in \mathbb{R}^{3 \times 2}
$$

This is read as:

> "**A is a 3 by 2 matrix.**"

Where:

* First number = rows
* Second number = columns

---

# Rows vs Columns

Consider:

$$
X =
\begin{bmatrix}
170 & 65 & 21 \
180 & 75 & 24 \
160 & 55 & 20
\end{bmatrix}
$$

Interpretation:

| Column | Meaning |
| ------ | ------- |
| 1      | Height  |
| 2      | Weight  |
| 3      | Age     |

Rows:

```text
Row 1 → Student 1
Row 2 → Student 2
Row 3 → Student 3
```

Columns:

```text
Column 1 → Heights
Column 2 → Weights
Column 3 → Ages
```

---

# Why This Matters in Machine Learning

Almost every dataset is stored as:

$$
X =
\begin{bmatrix}
\text{Feature 1} & \text{Feature 2} & \cdots & \text{Feature n}
\end{bmatrix}
$$

where:

* Each **row** is one training example.
* Each **column** is one feature.

For example:

| Size | Bedrooms | Price |
| ---: | -------: | ----: |
| 1200 |        2 |    45 |
| 1800 |        3 |    72 |
| 2400 |        4 |   105 |

The **feature matrix** is:

$$
X =
\begin{bmatrix}
1200 & 2 \
1800 & 3 \
2400 & 4
\end{bmatrix}
$$

The **target vector** is:

$$
y =
\begin{bmatrix}
45 \
72 \
105
\end{bmatrix}
$$

Notice that we separate **inputs** from **outputs**.

This is exactly how libraries like NumPy, Scikit-Learn, TensorFlow, and PyTorch expect data.

---

# Worked Examples

## Example 1

Matrix:

$$
A =
\begin{bmatrix}
1 & 2 \
3 & 4
\end{bmatrix}
$$

Questions:

* Rows? → **2**
* Columns? → **2**
* Shape? → **(2, 2)**

---

## Example 2

Matrix:

$$
B =
\begin{bmatrix}
5 & 8 & 1 \
9 & 4 & 7
\end{bmatrix}
$$

Questions:

* Rows? → **2**
* Columns? → **3**
* Shape? → **(2, 3)**

---

# Python Implementation

Creating a matrix:

```python
import numpy as np

X = np.array([
    [8, 95, 12],
    [6, 88, 10],
    [9, 97, 15]
])

print(X)
```

Output:

```text
[[ 8 95 12]
 [ 6 88 10]
 [ 9 97 15]]
```

---

## Shape

```python
print(X.shape)
```

Output:

```text
(3, 3)
```

Meaning:

* 3 rows
* 3 columns

---

## Accessing Rows

```python
print(X[0])
```

Output:

```text
[ 8 95 12]
```

The first student's feature vector.

---

## Accessing Columns

```python
print(X[:, 1])
```

Output:

```text
[95 88 97]
```

The attendance column.

Explanation:

* `:` means **all rows**.
* `1` means **second column** (remember, indexing starts at 0).

---

## Matrix Arithmetic

```python
A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

print(A + B)
```

Output:

```text
[[ 6  8]
 [10 12]]
```

NumPy adds corresponding elements.

---

# Common Mistakes

!!! warning "Common Mistakes"

```
1. Confusing rows with columns.
2. Thinking the shape `(3, 2)` means 3 columns and 2 rows—it means **3 rows and 2 columns**.
3. Mixing the feature matrix `X` with the target vector `y`.
4. Forgetting that Python indexing starts at `0`.
```

---

# Practice Problems

## Easy

1. How many rows and columns does the following matrix have?

$$
\begin{bmatrix}
1 & 2 & 3 \
4 & 5 & 6
\end{bmatrix}
$$

2. What is the shape of the matrix above?

3. In a machine learning dataset, what does **one row** usually represent?

---

## Medium

1. Create a `2 × 4` NumPy matrix.

2. Print the third column.

3. What is the difference between a row and a column in a dataset?

---

## Challenge

A dataset contains:

* 10,000 students
* 6 features per student

Answer:

1. What is the shape of the feature matrix?
2. How many rows?
3. How many columns?
4. If exam scores are stored separately, what is the shape of the target vector?

---

# Coding Assignment

Create a dataset of **five students** with these features:

* Hours Studied
* Attendance
* Assignments Completed

Store them in a NumPy matrix called `X`.

Then:

1. Print the matrix.
2. Print its shape.
3. Print the second student's data.
4. Print the attendance column.
5. Add `1` to every assignment count.

---

# Interview Questions

1. What is a matrix?
2. What is the difference between a vector and a matrix?
3. In machine learning, what does one row represent?
4. What does one column represent?
5. What does the shape `(1000, 20)` tell you about a dataset?
6. Why do machine learning libraries store data as matrices?

---

# Reflection

Before moving to **Dot Product**, ask yourself:

* Can I explain why a matrix is just a collection of vectors?
* Can I identify rows, columns, and shapes?
* Can I explain why datasets are stored as matrices?
* Can I comfortably access rows and columns using NumPy?

If the answer is **yes**, you're ready for the **Dot Product**—the operation that powers linear regression, neural networks, and nearly every prediction model you'll build from this point onward.

---

## Why the Dot Product Is Next

Everything we've learned has been building toward one question:

> **Given a student's features, how can a machine compute a prediction?**

The answer begins with the **dot product**.

The equation you'll soon understand,

$$
\hat{y} = Xw + b
$$

depends on one fundamental operation:

$$
Xw
$$

That operation is matrix–vector multiplication, built on the **dot product**. Once you understand it intuitively, the mathematics of machine learning becomes much more natural.

Excellent. This is the chapter where many students have their first **"aha!" moment** in Machine Learning.

Until now we've learned:

* A **vector** describes one object.
* A **matrix** stores many objects.

Now we'll answer the question:

> **How does a machine combine all the features of one object into a single prediction?**

The answer is the **dot product**.

---

# Machine Learning From First Principles

# Week 1 – Day 2

# Chapter 2 (Part 5): The Dot Product

> *"The dot product is the operation that transforms many features into one meaningful number."*

---

# Learning Objectives

By the end of this chapter, you will be able to:

* Explain what a dot product is.
* Understand why it is used in machine learning.
* Compute dot products by hand.
* Compute dot products using NumPy.
* Understand how the dot product becomes the foundation of prediction.

---

# Story-Based Introduction

## Hiring a New Employee

Imagine you're hiring a software engineer.

You decide to evaluate each candidate using four criteria:

| Feature           | Importance |
| ----------------- | ---------: |
| Programming Skill |          5 |
| Communication     |          2 |
| Problem Solving   |          4 |
| Experience        |          3 |

Notice something interesting.

Not every feature is equally important.

Programming skill matters more than communication.

Problem solving matters more than experience.

Now suppose Candidate A has the following scores:

| Feature         | Score |
| --------------- | ----: |
| Programming     |     9 |
| Communication   |     8 |
| Problem Solving |    10 |
| Experience      |     6 |

How do we combine these four numbers into **one overall score**?

We shouldn't simply add them because programming is more important than communication.

Instead, we **weight** each feature by its importance.

That weighted combination is exactly what the **dot product** computes.

---

# Why Was the Dot Product Invented?

Suppose we want to predict the price of a house.

The features are:

* Size
* Bedrooms
* Bathrooms
* Age

Each feature influences the price differently.

A larger house might add a lot to the price.

An extra bedroom might add a moderate amount.

An older house might reduce the price.

We need a way to combine all of these effects into **one prediction**.

The dot product gives us that mechanism.

---

# Intuition

Think of cooking.

Suppose you're making lemonade.

Ingredients:

* Water
* Sugar
* Lemon juice

Each ingredient contributes differently to the final taste.

You don't just count the ingredients.

You decide **how much** of each ingredient to use.

Machine learning does the same thing.

Features are the ingredients.

Weights decide how important each ingredient is.

The dot product mixes them into a single result.

---

# Visual Explanation

Suppose we have:

Student features:

$$
\mathbf{x} =
\begin{bmatrix}
8 \
95 \
12
\end{bmatrix}
$$

Model weights:

$$
\mathbf{w} =
\begin{bmatrix}
4 \
0.3 \
2
\end{bmatrix}
$$

The dot product pairs each feature with its corresponding weight:

```text
Hours Studied      × Weight
      8            ×   4

Attendance         × Weight
      95           × 0.3

Assignments        × Weight
      12           ×   2
```

Then it adds all of the results together.

---

# Feynman Explanation

Imagine explaining this to a 10-year-old.

Suppose your teacher says:

> Homework counts twice as much as attendance.

Now your report card isn't just adding marks.

Some things matter more than others.

The teacher multiplies each score by its importance and then adds everything together.

That is exactly what the dot product does.

---

# Mathematical Foundations

Suppose

$$
\mathbf{x} =
\begin{bmatrix}
x_1 \
x_2 \
x_3
\end{bmatrix}
$$

and

$$
\mathbf{w} =
\begin{bmatrix}
w_1 \
w_2 \
w_3
\end{bmatrix}
$$

Their dot product is

$$
\mathbf{x}\cdot\mathbf{w}
=========================

x_1w_1+x_2w_2+x_3w_3
$$

### What do the symbols mean?

* $\mathbf{x}$ = feature vector
* $\mathbf{w}$ = weight vector
* $x_i$ = the $i^{th}$ feature
* $w_i$ = the weight for the $i^{th}$ feature

Each feature is multiplied by **its own corresponding weight**, and all those products are summed into a single number.

---

# Worked Example 1

Student:

$$
\mathbf{x} =
\begin{bmatrix}
8 \
95 \
12
\end{bmatrix}
$$

Weights:

$$
\mathbf{w} =
\begin{bmatrix}
4 \
0.3 \
2
\end{bmatrix}
$$

Step 1:

Multiply corresponding entries.

$$
8\times4=32
$$

$$
95\times0.3=28.5
$$

$$
12\times2=24
$$

Step 2:

Add them.

$$
32+28.5+24=84.5
$$

The dot product equals

$$
84.5
$$

---

# Worked Example 2

$$
\mathbf{a}=
\begin{bmatrix}
2\
3\
4
\end{bmatrix}
$$

$$
\mathbf{b}=
\begin{bmatrix}
5\
6\
7
\end{bmatrix}
$$

Compute:

$$
2\times5
+
3\times6
+
4\times7
$$

$$
10+18+28=56
$$

Answer:

$$
56
$$

---

# Why This Is So Important

Remember Day 1?

We said the goal of machine learning is to predict something.

The prediction equation for linear regression is

$$
\hat{y}=Xw+b
$$

Let's focus on

$$
Xw
$$

Suppose

$$
X=
\begin{bmatrix}
8 & 95 & 12
\end{bmatrix}
$$

and

$$
w=
\begin{bmatrix}
4\
0.3\
2
\end{bmatrix}
$$

The multiplication

$$
Xw
$$

is simply a **dot product**.

That means every prediction in linear regression starts by computing a weighted sum of the features.

This is why the dot product is one of the most fundamental operations in machine learning.

---

# Python Implementation

Using NumPy:

```python
import numpy as np

x = np.array([8, 95, 12])
w = np.array([4, 0.3, 2])

prediction = np.dot(x, w)

print(prediction)
```

Output:

```text
84.5
```

---

## Using the `@` Operator

NumPy also supports matrix multiplication using `@`:

```python
prediction = x @ w

print(prediction)
```

Output:

```text
84.5
```

Both approaches produce the same result for 1D vectors.

---

# Common Mistakes

!!! warning "Common Mistakes"

```
1. Multiplying every feature by every weight. Each feature should only be multiplied by its corresponding weight.
2. Forgetting to add the products after multiplying.
3. Taking the dot product of vectors with different lengths.
4. Confusing element-wise multiplication (`x * w`) with the dot product (`np.dot(x, w)`).
```

---

# Practice Problems

## Easy

1. Compute the dot product:

$$
\begin{bmatrix}
1\
2\
3
\end{bmatrix}
\cdot
\begin{bmatrix}
4\
5\
6
\end{bmatrix}
$$

2. Can you compute the dot product of vectors with different numbers of components? Why?

3. In machine learning, what do the **weights** represent?

---

## Medium

1. Compute the dot product by hand:

$$
\begin{bmatrix}
5\
2\
7
\end{bmatrix}
\cdot
\begin{bmatrix}
3\
4\
1
\end{bmatrix}
$$

2. Write a NumPy program that computes the dot product of:

```python
x = np.array([3, 6, 9])
w = np.array([2, 5, 1])
```

3. What is the difference between:

```python
x * w
```

and

```python
np.dot(x, w)
```

---

## Challenge

A house is described by the feature vector:

$$
\mathbf{x}=
\begin{bmatrix}
2000\
3\
10
\end{bmatrix}
$$

where the features are:

* Size (sq ft)
* Bedrooms
* Age (years)

The model has learned the weights:

$$
\mathbf{w}=
\begin{bmatrix}
120\
15000\
-2000
\end{bmatrix}
$$

1. Compute the dot product by hand.
2. Explain why the third weight is negative.
3. Which feature contributes the most to the prediction?

---

# Coding Assignment

Using NumPy:

1. Create two vectors of length **5**.
2. Compute their dot product using `np.dot()`.
3. Compute the same result using the `@` operator.
4. Verify that both answers are identical.
5. Compute the element-wise multiplication using `*`.
6. Explain, in your own words, why the result of `*` is different from the dot product.

---

# Interview Questions

1. What is a dot product?
2. Why is the dot product important in machine learning?
3. What do the weights represent?
4. Can vectors of different lengths have a dot product?
5. What is the difference between element-wise multiplication and the dot product?
6. How does the dot product relate to the prediction equation $\hat{y} = Xw + b$?

---

# Reflection

Before moving to **Broadcasting**, ask yourself:

* Can I compute a dot product by hand?
* Do I understand why each feature has a corresponding weight?
* Can I explain the difference between `x * w` and `np.dot(x, w)`?
* Can I see why the dot product naturally produces a single prediction?

If you can answer **yes**, you've just learned one of the most important mathematical operations in machine learning. The next topic—**Broadcasting**—will show you how NumPy performs operations on arrays of different shapes efficiently, a feature you'll use constantly when implementing ML algorithms.

Excellent. This is the final major topic of **Week 1 – Day 2**.

Many beginners think broadcasting is a "NumPy trick."

It isn't.

Broadcasting is one of the reasons NumPy is so powerful. Once you understand it, you'll see it used everywhere—in **Linear Regression, Neural Networks, TensorFlow, PyTorch, and almost every ML library**.

---

# Machine Learning From First Principles

# Week 1 – Day 2

# Chapter 2 (Part 6): Broadcasting

> *"Broadcasting lets NumPy perform operations between arrays of different shapes without manually copying data."*

---

# Learning Objectives

By the end of this chapter, you will be able to:

* Explain why broadcasting exists.
* Understand the intuition behind broadcasting.
* Know the broadcasting rules.
* Predict when broadcasting will succeed or fail.
* Use broadcasting in NumPy programs.
* Recognize where broadcasting is used in machine learning.

---

# Story-Based Introduction

## The Teacher and the Bonus Marks

A teacher has the marks of five students stored in a table.

| Student | Math | Science | English |
| ------- | ---- | ------- | ------- |
| A       | 70   | 75      | 80      |
| B       | 85   | 90      | 88      |
| C       | 60   | 65      | 70      |
| D       | 95   | 92      | 96      |
| E       | 78   | 80      | 82      |

One day, the teacher decides:

> "Everyone gets **5 bonus marks**."

There are two ways to do this.

### Method 1

Go through every subject of every student and add 5 manually.

Very slow.

### Method 2

Tell the computer:

```python
marks + 5
```

NumPy automatically understands:

> "Add 5 to every element."

You never wrote a loop.

This automatic expansion is called **broadcasting**.

---

# Why Was Broadcasting Invented?

Imagine a dataset with:

* 10 million students
* 50 features each

Without broadcasting, you would need nested loops.

```python
for every student:
    for every feature:
        ...
```

Besides being longer to write, explicit Python loops are much slower than NumPy's optimized operations.

Broadcasting lets NumPy apply the operation internally, using highly optimized code.

---

# Intuition

Imagine a teacher announcing:

> "Everyone in this classroom gets one extra notebook."

The teacher doesn't hand each notebook individually in your mental model.

The instruction applies to **everyone**.

Broadcasting works the same way.

A single value can behave **as if** it were repeated across an entire array.

The important point is that NumPy **acts as though** the value is copied—it doesn't necessarily create millions of actual copies in memory.

---

# Visual Explanation

Suppose we have:

```text
Marks

[70 80 90]
```

Add:

```text
5
```

NumPy behaves **as if** it first created:

```text
[5 5 5]
```

Then performs:

```text
[70 80 90]
+
[ 5  5  5]
---------
[75 85 95]
```

Notice that you only wrote:

```python
marks + 5
```

The expansion happens automatically.

---

# Feynman Explanation

Imagine explaining broadcasting to a younger sibling.

You could say:

> Suppose I tell everyone in your class to stand up.

I don't need to point to each student one by one.

One instruction applies to everyone.

Broadcasting is the same idea.

One value is applied across many values automatically.

---

# Mathematical Foundations

Suppose

$$
\mathbf{x} =
\begin{bmatrix}
70\
80\
90
\end{bmatrix}
$$

and we compute

$$
\mathbf{x}+5
$$

Mathematically, NumPy behaves as though we had written

$$
\begin{bmatrix}
70\
80\
90
\end{bmatrix}
+
\begin{bmatrix}
5\
5\
5
\end{bmatrix}
=============

\begin{bmatrix}
75\
85\
95
\end{bmatrix}
$$

Remember:

> NumPy **behaves as if** the scalar becomes a vector.

It does **not** literally allocate a new vector filled with 5s in memory.

---

# Broadcasting Rules

Broadcasting follows simple rules.

Starting from the **last dimension**, NumPy compares the shapes.

Two dimensions are compatible if:

1. They are equal.
2. One of them is `1`.

Otherwise,

❌ Broadcasting fails.

---

## Example 1

Shape:

```text
(3,)
```

*

Scalar

```text
()
```

Result:

✅ Works

---

## Example 2

```text
(3,3)

+

(3,)
```

Works.

Visualization:

```text
1 2 3
4 5 6
7 8 9

+

10 20 30

↓

11 22 33
14 25 36
17 28 39
```

The row vector is applied to every row.

---

## Example 3

```text
(3,2)

+

(2,)
```

Works.

Each row receives the two values.

---

## Example 4

```text
(3,2)

+

(3,)
```

Fails.

Why?

Compare from the last dimension:

```text
(3,2)
   ↑

(3,)
 ↑
```

Last dimensions:

```text
2

vs

3
```

Not equal.

Neither is 1.

NumPy raises an error.

---

# Worked Examples

## Example 1

```python
import numpy as np

marks = np.array([70, 80, 90])

print(marks + 5)
```

Output

```text
[75 85 95]
```

---

## Example 2

```python
A = np.array([
    [1,2,3],
    [4,5,6]
])

B = np.array([10,20,30])

print(A + B)
```

Output

```text
[[11 22 33]
 [14 25 36]]
```

Broadcasting applied `B` to every row.

---

## Example 3

```python
A = np.array([
    [1,2],
    [3,4]
])

B = np.array([10,20,30])

print(A + B)
```

Output

```text
ValueError:
operands could not be broadcast together
```

This happens because the shapes are incompatible.

---

# Why Broadcasting Matters in Machine Learning

Suppose your prediction equation is

$$
\hat{y}=Xw+b
$$

Here:

* $Xw$ produces one prediction for each training example.
* $b$ is a **single number** called the **bias**.

If there are five predictions,

$$
\begin{bmatrix}
82\
75\
91\
88\
79
\end{bmatrix}
+
3
$$

NumPy broadcasts the bias:

$$
\begin{bmatrix}
82\
75\
91\
88\
79
\end{bmatrix}
+
\begin{bmatrix}
3\
3\
3\
3\
3
\end{bmatrix}
=============

\begin{bmatrix}
85\
78\
94\
91\
82
\end{bmatrix}
$$

This is exactly how we'll implement Linear Regression tomorrow.

---

# Python Implementation

### Scalar Broadcasting

```python
import numpy as np

x = np.array([10, 20, 30])

print(x + 5)
```

---

### Row Broadcasting

```python
A = np.array([
    [1,2,3],
    [4,5,6]
])

B = np.array([10,20,30])

print(A + B)
```

---

### Column Broadcasting

```python
A = np.array([
    [1,2],
    [3,4],
    [5,6]
])

B = np.array([
    [10],
    [20],
    [30]
])

print(A + B)
```

Output

```text
[[11 12]
 [23 24]
 [35 36]]
```

Notice the shape of `B` is `(3, 1)`.

It is broadcast across the columns.

---

# Common Mistakes

!!! warning "Common Mistakes"

```
1. Thinking broadcasting always works.
2. Forgetting that NumPy compares shapes from the **last dimension**.
3. Assuming broadcasting physically copies the data.
4. Confusing row vectors `(3,)` with column vectors `(3,1)`.
5. Ignoring the shape of arrays before performing operations.
```

---

# Practice Problems

## Easy

1. Predict the output of:

```python
np.array([2,4,6]) + 3
```

2. Will `(4,)` and `()` broadcast successfully? Why?

3. What does broadcasting save us from writing?

---

## Medium

1. What is the result of:

```python
A = np.array([
    [1,2],
    [3,4]
])

B = np.array([10,20])

A + B
```

2. Will `(2,3)` and `(3,)` broadcast? Explain.

3. Will `(2,3)` and `(2,)` broadcast? Why or why not?

---

## Challenge

Without running Python, determine whether each operation succeeds or fails.

1.

```text
(5,4)
+
(4,)
```

2.

```text
(5,4)
+
(5,1)
```

3.

```text
(5,4)
+
(5,)
```

For each one:

* State **Works** or **Fails**.
* Explain your reasoning using the broadcasting rules.

---

# Coding Assignment

Create the following matrix:

```python
X = np.array([
    [10,20,30],
    [40,50,60],
    [70,80,90]
])
```

Then:

1. Add `5` to every element.
2. Add the row vector:

```python
np.array([1,2,3])
```

3. Add the column vector:

```python
np.array([
    [10],
    [20],
    [30]
])
```

4. Print the shape of every array before performing the operations.
5. Explain **why broadcasting succeeds** in each case.

---

# Interview Questions

1. What is broadcasting in NumPy?
2. Why was broadcasting introduced?
3. What are the two broadcasting compatibility rules?
4. Does broadcasting physically copy data in memory?
5. Why is broadcasting useful in machine learning?
6. Give an example where broadcasting fails.

---

# Reflection

Before ending **Week 1 – Day 2**, ask yourself:

* Can I explain broadcasting without mentioning NumPy documentation?
* Can I predict when broadcasting will succeed or fail by comparing shapes?
* Do I understand the difference between scalar, row, and column broadcasting?
* Can I explain why broadcasting is essential for equations like

$$
\hat{y} = Xw + b
$$

If the answer is **yes**, you've completed one of the most important foundations for implementing machine learning algorithms. Tomorrow, we'll use everything you've learned—vectors, matrices, dot products, and broadcasting—to build your first machine learning model: **Linear Regression** from first principles.
