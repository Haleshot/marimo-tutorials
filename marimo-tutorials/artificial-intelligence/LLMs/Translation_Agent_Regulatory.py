import marimo

__generated_with = "0.9.12"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import os
    return SmoothingFunction, mo, nltk, os, pd, sentence_bleu


@app.cell
def __(mo, os):
    os_key = os.environ.get("OPENAI_API_KEY")
    input_key = mo.ui.text(label="OpenAI API key", kind="password")
    input_key if not os_key else None

    input_key
    return input_key, os_key


@app.cell
def __(input_key, mo, os_key):
    openai_key = os_key or input_key.value

    mo.stop(
            not openai_key,
            mo.md("Please set the OPENAI_API_KEY environment variable or provide it in the input field"),
        )
    return (openai_key,)


@app.cell
def __(mo):
    mo.md(r"""#A Regulatory and Translating Agent""")
    return


@app.cell
def __(mo, openai_key):
    Regulatory_Assistant = mo.ui.chat(
        mo.ai.llm.openai(
            "gpt-4o",
            system_message="You are a regulatory assistant at a biotech firm. You understand English, and you communicate solely in French (Canadian). You are conise.",
            api_key= openai_key
        ),
        show_configuration_controls=True
    )
    Regulatory_Assistant
    return (Regulatory_Assistant,)


@app.cell
def __(Regulatory_Assistant, mo, pd):
    Log = pd.DataFrame(Regulatory_Assistant.value)
    mo.ui.table(Log)
    return (Log,)


@app.cell
def __(SmoothingFunction, sentence_bleu):
    def calculate_bleu(reference, candidate):
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
        return bleu_score
    return (calculate_bleu,)


@app.cell
def __(Log, Reference_Response, calculate_bleu, nltk):
    reference_sentence = Reference_Response.lower()
    candidate_sentence = Log['content'][1].lower()

    # Tokenizing sentences
    reference = [nltk.word_tokenize(reference_sentence)]  # reference should be a list of lists
    candidate = nltk.word_tokenize(candidate_sentence)

    # Calculating BLEU score
    bleu_score = calculate_bleu(reference, candidate)
    return (
        bleu_score,
        candidate,
        candidate_sentence,
        reference,
        reference_sentence,
    )


@app.cell
def __(Reference):
    Reference_Response = Reference.value
    return (Reference_Response,)


@app.cell
def __(Log, mo):
    Row_Number = mo.ui.number(1,len(Log),2,label = 'Row Number')
    Reference = mo.ui.text(label='Reference',placeholder='Original translation')

    mo.hstack([Reference,Row_Number])
    return Reference, Row_Number


@app.cell
def __(bleu_score, mo):
    mo.md(rf"""
    <ul>
    <li> BLEU Score: {bleu_score} </li> 
    </ul>
    """)
    return


if __name__ == "__main__":
    app.run()
