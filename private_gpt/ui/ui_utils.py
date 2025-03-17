import gradio as gr


def chat_all(msg, history1, history2, history3):

    return [[
        history1 + [[msg, msg]],
        history2 + [[msg, msg]],
        history3 + [[msg, msg]],
    ]]