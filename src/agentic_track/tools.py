"""The agent's tools + the dispatch registry that maps names to functions."""


def calculator(expression):
    return str(eval(expression))


def word_count(text):
    return str(len(text.split()))


TOOLS = {
    "calculator": calculator,
    "word_count": word_count,
}