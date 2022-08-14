# Chapter 2
## A Crash Course in Python

### Zen of Python
- "There should be one -- and preferably only one -- obvious way to do it"
- "Beautiful is better than ugly. Explicit is better than implicit. Simple is
  better than complex."

### White Space Formatting
- White space is very important in python
- Indentation with tabs vs spaces can screw with syntax
- use a backslash for line continuation
    ```
    two_plus_three = 2 + \
                     3
    ```

### Python Functions
- functions are first class, which means they can be assigned to variables and
  passed to other functions as arguments
- lambda functions
    - can be assigned to vars, but should probably just be a `def` function
        ```
        double = lambda x: 2 * x
        ```

### Lists
- List slice syntax `list[i:j]` i is *inclusive* j is *not inclusive*
- Third argument in slice notation is the step
    - can be negative to traverse a list backwards
- Variable unpacking
    - `x, y = [1, 2]`
    - `_, y = [1,2]` ignores first element

### Dictionaries
- See example code for dictionary methods
- Note `<variable> in dict.values()` does not traverse values that are lists.

### Counters
- Turns sequence of values into a defaultdict(int)-like object mapping

### Sets
- List of unique values
- Much faster to iterate a set vs a list or dict
- Easiest way to find distinct values in a collection

### Truthiness
-`all()` function will return if all elements in a list are truthy


### Assert statements
- Assert statements can autoprint a message on failure
    ```
    assert 1 + 1 == 2, "1 +1 should equal 2 but it didnt"
    ```
- Can also be used to check on inputs for functions
    ```
    def smalelst_item(xs):
        assert xs, "empty list has no smallest item"
        return min(xs)
    ```

### Iterables and Generators
- You can create generators using comprehension syntax
    ```
     evens_below_20 = (i for i in generate_range(20 ) if i % 2 == 0)
    ```
- Parenthesis denote that the above line is a generator comprehension

### Randomness
- To get a random sample of items from a list without duplicates
    ```
    random.sample(list)
    ```
