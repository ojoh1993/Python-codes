#!/usr/bin/python
# -*- coding: UTF-8 -*-

def factorial(number):

    # error handling
    if not isinstance(number, int):
        raise TypeError("Sorry. 'number' must be an integer.")
    if not number >= 0:
        raise ValueError("Sorry. 'number' must be zero or positive.")

    def inner_factorial(number):
        if number <= 1:
            return 1
        return number*inner_factorial(number-1)
    return inner_factorial(number)


# call the outer function
print(factorial(4))

student_tuples = [
    ('john', 'A', 15.2),
    ('jane', 'B', 12.9),
    ('dave', 'B', 10.8),
    ]

student_tuples = sorted(student_tuples, key=lambda student: student[2])
print(student_tuples[0])
