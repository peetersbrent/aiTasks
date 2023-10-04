import streamlit as st
from simpleai.search import CspProblem, backtrack

st.title("Task 1 AI - Brent Peeters")

word_1 = st.text_input("Enter the first value: ").upper()
operator = st.text_input("Enter the operator: ")
word_2 = st.text_input("Enter the second value: ").upper()
sum_1 = st.text_input("Enter the third value: ").upper()

if st.button("Los de som op!"):

    combine_string = word_1 + word_2 + sum_1

    variables = set(combine_string)

    domains = {}

    for letter in set(combine_string):
        if letter in (word_1[0], word_2[0], sum_1[0]):
            if letter.isalpha():  # Check if the character is a letter
                domain = list(range(1, 10))
        else:
            if letter.isalpha():  # Check if the character is a letter
                domain = list(range(0, 10))
        domains[letter] = domain

    def constraint_unique(variables, values):
        return len(values) == len(set(values))  # remove repeated values and count

    def constraint_add(variables, values):
        word1 = "";
        for letter in word_1:
            index = variables.index(letter)
            word1 += str(values[index])

        word2 = "";
        for letter in word_2:
            index = variables.index(letter)
            word2 += str(values[index])

        calculated = "";
        for letter in sum_1:
            index = variables.index(letter)
            calculated += str(values[index])

        if (operator == "+"):
            return (int(word1) + int(word2)) == int(calculated)
        elif (operator == "-"):
            return (int(word1) - int(word2)) == int(calculated)
        elif (operator == "*"):
            return (int(word1) * int(word2)) == int(calculated)
        else:
            return (int(word1) / int(word2)) == int(calculated)

    constraints = [
        (variables, constraint_unique),
        (variables, constraint_add),
    ]

    problem = CspProblem(variables, domains, constraints)

    output = backtrack(problem)

    result1 = ''.join([str(output[letter]) for letter in word_1])
    result2 = ''.join([str(output[letter]) for letter in word_2])

    if (operator == "+"):
        st.text(word_1)
        st.text(" +")
        st.text(word_2)
        st.text(" =")
        st.text(sum_1)
    elif (operator == "-"):
        st.text(word_1)
        st.text(" -")
        st.text(word_2)
        st.text(" =")
        st.text(sum_1)
    elif (operator == "*"):
        st.text(word_1)
        st.text(" *")
        st.text(word_2)
        st.text(" =")
        st.text(sum_1)
    else:
        st.text(word_1)
        st.text(" /")
        st.text(word_2)
        st.text(" =")
        st.text(sum_1)

    st.text("----------------------------")

    plus = int(result1) + int(result2)
    min = int(result1) - int(result2)
    maal = int(result1) * int(result2)
    gedeeld = int(result1) / int(result2)

    if (operator == "+"):
        st.text(result1)
        st.text(" +")
        st.text(result2)
        st.text(" =")
        st.text(plus)
    elif (operator == "-"):
        st.text(result1)
        st.text(" -")
        st.text(result2)
        st.text(" =")
        st.text(min)
    elif (operator == "*"):
        st.text(result1)
        st.text(" *")
        st.text(result2)
        st.text(" =")
        st.text(maal)
    else:
        st.text(result1)
        st.text(" /")
        st.text(result2)
        st.text(" =")
        st.text(gedeeld)
    # print('\nSolutions:', output)
