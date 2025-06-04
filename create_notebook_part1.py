import nbformat

# Create a new notebook object
notebook = nbformat.v4.new_notebook()

# --- Overall Notebook Title and Introduction ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""# A Beginner's Journey into Data Science

This Jupyter Notebook is designed to be a gentle introduction to the fascinating world of Data Science. We'll cover foundational concepts, essential Python skills for data tasks, descriptive statistics, basic data visualization, and a peek into machine learning principles, all while utilizing functions from our custom `pymath` library where applicable."""
))

# --- Part 1: Foundations ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""## Part 1: Foundations"""
))

# --- Section 1: Welcome to the World of Data Science! ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 1: Welcome to the World of Data Science!"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Understand what data science is.
    *   Learn about the typical lifecycle of a data science project.
    *   Become aware of ethical considerations in data science."""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **What is Data Science?**
        *   Definition and its interdisciplinary nature.
        *   Impact of data science across various industries."""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Key Components of Data Science:**
        *   Statistics: The science of learning from data.
        *   Programming: Tools to manipulate and analyze data (Python focus).
        *   Domain Expertise: Understanding the context of the data."""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **The Data Science Lifecycle:**
        *   Business Understanding / Problem Definition
        *   Data Collection: Gathering relevant data.
        *   Data Cleaning/Preprocessing: Handling missing values, errors, and formatting.
        *   Exploratory Data Analysis (EDA): Uncovering patterns and insights.
        *   Model Building: Selecting and training machine learning models.
        *   Model Evaluation: Assessing model performance.
        *   Deployment/Interpretation: Using the model and communicating results."""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Ethical Considerations in Data Science:**
        *   Bias: How data or algorithms can lead to unfair outcomes.
        *   Privacy: Protecting sensitive information.
        *   Transparency: Understanding how models make decisions."""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **`pymath` library usage:** None in this section."""
))


# --- Section 2: Python Essentials for Data Exploration ---
notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""### Section 2: Python Essentials for Data Exploration"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Learning Objectives:**
    *   Refresh basic Python syntax relevant to data tasks.
    *   Understand how to use lists and dictionaries for data manipulation.
    *   Learn to write simple functions."""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **Content:**
    *   **Quick Python Recap:**
        *   Variables and Data Types: Integers, floats, strings, booleans.
        *   Basic Operators:
            *   Arithmetic (`+`, `-`, `*`, `/`, `%`, `**`)
            *   Comparison (`==`, `!=`, `>`, `<`, `>=`, `<=`)
            *   Logical (`and`, `or`, `not`)"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Variables & Basic Data Types"""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Integers
a = 10
print(f"Integer: {a}, Type: {type(a)}")

# Floats
b = 3.14
print(f"Float: {b}, Type: {type(b)}")

# Strings
c = "Hello, Data Science!"
print(f"String: {c}, Type: {type(c)}")

# Booleans
d = True
e = False
print(f"Boolean d: {d}, Type: {type(d)}")
print(f"Boolean e: {e}, Type: {type(e)}")"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Operators"""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Arithmetic Operators
x = 10
y = 3
print(f"{x} + {y} = {x + y}")
print(f"{x} - {y} = {x - y}")
print(f"{x} * {y} = {x * y}")
print(f"{x} / {y} = {x / y}") # Float division
print(f"{x} // {y} = {x // y}") # Integer division
print(f"{x} % {y} = {x % y}")   # Modulus
print(f"{x} ** {y} = {x ** y}") # Exponentiation

# Comparison Operators
print(f"Is {x} > {y}? {x > y}")
print(f"Is {x} == {y}? {x == y}")

# Logical Operators
p = True
q = False
print(f"{p} and {q} = {p and q}")
print(f"{p} or {q} = {p or q}")
print(f"not {p} = {not p}")"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Control Flow:**
        *   `if/else` statements for conditional logic.
        *   `for` loops for iterating over sequences.
        *   `while` loops for repeated execution based on a condition."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# if/else
grade = 85
if grade >= 90:
    print("A")
elif grade >= 80:
    print("B")
else:
    print("C or lower")

# for loop (conceptual pymath usage for factorial)
# Assuming pymath.lib.math is accessible, otherwise this is conceptual.
# from pymath.lib.math import factorial
# print("Factorials from 0 to 4:")
# for i in range(5):
#     # print(f"Factorial of {i} is {factorial(i)}") # Conceptual
#     print(f"Factorial of {i} is (conceptual pymath.factorial({i}))")


# while loop
count = 0
while count < 3:
    print(f"Count is {count}")
    count += 1"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Core Data Structures:**
        *   Lists
        *   Dictionaries"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Lists"""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# List creation
my_list = [1, 2, 3, 'apple', 3.14, True]
print(f"Original list: {my_list}")

# Indexing
print(f"First element: {my_list[0]}")
print(f"Last element: {my_list[-1]}")

# Slicing
print(f"Slice [1:3]: {my_list[1:3]}")

# Methods
my_list.append('banana')
print(f"After append: {my_list}")
print(f"Length of list: {len(my_list)}")

numbers = [3, 1, 4, 1, 5, 9, 2, 6]
numbers.sort()
print(f"Sorted numbers: {numbers}")"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""#### Dictionaries"""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Dictionary creation
my_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}
print(f"Dictionary: {my_dict}")

# Accessing values
print(f"Name: {my_dict['name']}")
print(f"Age using get(): {my_dict.get('age')}")

# Methods
print(f"Keys: {my_dict.keys()}")
print(f"Values: {my_dict.values()}")
print(f"Items: {my_dict.items()}")

my_dict['occupation'] = 'Engineer' # Adding a new key-value pair
print(f"Updated dictionary: {my_dict}")"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""    *   **Functions:**
        *   Defining functions using `def`.
        *   Parameters and arguments.
        *   `return` statement for output.
        *   Docstrings for explaining function purpose."""
))
notebook.cells.append(nbformat.v4.new_code_cell(
    source="""# Defining a function
def greet(name):
    \"\"\"This function greets the person passed in as a parameter.\"\"\"
    return f"Hello, {name}!"

print(greet("Data Scientist"))

# Function with conceptual pymath usage
# Assuming pymath.lib.math is accessible
# from pymath.lib.math import is_prime

def display_number_properties(num):
    \"\"\"Displays properties of a number, conceptually using pymath.\"\"\"
    # prop_is_prime = is_prime(num) # Conceptual
    prop_is_prime = f"(conceptual pymath.is_prime({num}))"

    if num % 2 == 0:
        parity = "even"
    else:
        parity = "odd"
    return f"Number {num} is {parity}. Is it prime? {prop_is_prime}."

print(display_number_properties(7))
print(display_number_properties(10))"""
))

notebook.cells.append(nbformat.v4.new_markdown_cell(
    source="""*   **`pymath` library usage:**
    *   Example: Loop to calculate `factorial(n)` for `n` from 0 to 5 using `pymath.lib.math.factorial`. (Conceptual, shown in Control Flow)
    *   Example: Define a simple Python function that internally calls a basic `pymath` function (e.g., `is_prime`) and returns a formatted string. (Conceptual, shown in Functions)"""
))


# Write the notebook to a file
with open('data_science_learning_notebook.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(notebook, f)

print("Notebook Part 1 created successfully.")
