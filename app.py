import csv
import gradio as gr
from huggingface_hub import InferenceClient
import os

from matplotlib import colors
from rag import run_rag
from gradio.themes.utils import (
    colors,
    fonts,
    get_matching_version,
    get_theme_assets,
    sizes,
)
 
#  ================================================================================================================================
TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta" , token=TOKEN)
system_message ="You are a capable and freindly assistant."

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)
# ================================================================================================================================
class Int_State:
    def __init__(self):
        # initialise history of type list[tuple[str, str]]
        self.history = []
        self.current_query = ""
        self.current_response = ""
        self.roles = ["user", "system"]
        print("State has been initialise")
    
    def save_question(self, question):
        self.current_query = question
        self.current_response = ""
        self.history.append({"role": "user", "content": question})
        print("Question added ")
 
    def save_response(self,  assistant_message):
        # current_question = self.current_query
        self.current_response = assistant_message
        self.history.append({"role": "system", "content": assistant_message})
        print("Response saved  ")
 
 
    def get_history(self):
        return self.history
  
#  ================================================================================================================================
state = Int_State()
#  ================================================================================================================================
def clear_chat(chatbot ):
    state.history = []
 
    chatbot.clear() 
 
    yield ("" , chatbot) + (enable_btn,) * 5

#  ================================================================================================================================
def save_chat( question, answer, upvote, downvote, flag):
    file_path = "chat_data.csv"
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Add new row with provided data
    new_row = [question, answer, upvote, downvote, flag]
    data.append(new_row)

    # Write updated data back to CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("New row added successfully to", file_path)



def upvote_last_response():
    print("Upvoted")
    save_chat(state.current_query, state.current_response, 1, 0, 0)
    return (disable_btn,) * 3 + (enable_btn,)*2

def downvote_last_response():
    print("Downvoted")
    save_chat(state.current_query, state.current_response, 0, 1, 0)
    return (disable_btn,) * 3 + (enable_btn,)*2

def flag_last_response():
    print("Flagged")
    save_chat(state.current_query, state.current_response, 0, 0, 1)
    return (disable_btn,) * 3 + (enable_btn,)*2
 
def remove_last_response(chatbot): 
    print("Regenerated")
    textbox =state.current_query
    state.history.pop()
    state.history.pop()
    chatbot.clear()
    
    return (textbox ,chatbot ) + (enable_btn,) * 5
def quit_chat():
    return demo.close()

#  ================================================================================================================================ 

def chat(
    chatbot,
    message,
    max_tokens,
    temperature,
    top_p,
):
    question= message
    chatbot.append((question,""))
    yield ("" , chatbot) + (disable_btn,) * 5
    messages = [{"role": "system", "content": system_message}]
    history= state.get_history()
    state.save_question(message)

    for val in history:
        messages.append(val)

    messages.append({"role": "user", "content": run_rag(message)})
    response = "This is a response to the question"
    chatbot.append((question,""))
    for msg in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
       
        token = msg.choices[0].delta.content
        response += str(token)
        # chatbot.append(( response, response))
        # yield "" , chatbot 
    chatbot.clear()
    chatbot.append((question , response))
    state.save_response(response)
    yield ("" , chatbot) + (enable_btn,) * 5

 
  

#  ================================================================================================================================

theme = gr.themes.Base(
    primary_hue=colors.emerald,
    secondary_hue=colors.cyan,
    neutral_hue=colors.stone, 
    radius_size=sizes.radius_lg,
    spacing_size=sizes.spacing_sm,
    font=[gr.themes.GoogleFont('Poppins'), gr.themes.GoogleFont('Reddit Sans'), 'system-ui', 'sans-serif'],
)
EXAMPLES = [
    [ "Tell me about the latest news in the world ?"],
    [ "Tell me about the increase in the price of Bitcoin ?"],
    [ "Tell me about the actual situation in Ukraine ?"],
    [ "Tell me about current situation in palestine ?"],
]
 
#  ================================================================================================================================
 

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""
#  ================================================================================================================================
textbox = gr.Textbox(show_label=False,
                        placeholder="Enter a question or message...",
                        container=False,
                        show_copy_button=True
                        )

with gr.Blocks(title="RAG", theme=theme, css=block_css , fill_height=True) as demo:
    
    gr.Markdown("# **Retrieval Augmented Generation (RAG) Chatbot**" )
    gr.Markdown("This is a demo of a chatbot that uses the RAG system to generate responses to user queries. RAG is a combination of a retriever and a generator, which allows it to generate responses based on the context of the conversation. The chatbot can be used to answer questions, provide information, and engage in conversation with users.")
    with gr.Row(variant="panel"):
 
            
   
        with gr.Column(scale=10):
            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="Retrieval Augmented Generation (RAG) Chatbot",
                height=300,
                layout="bubble",
                min_width=1200,
                show_copy_button=True,
                show_share_button=True,
                placeholder="Ask a question or type a message...",
            )
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()

            with gr.Column(scale=1, min_width=100):
                submit_btn = gr.Button(value="Submit", variant="primary", interactive=True)
    
        
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn = gr.Button(value="👍  Upvote", interactive=False , variant="secondary")
                downvote_btn = gr.Button(value="👎  Downvote", interactive=False , variant="secondary")
                flag_btn = gr.Button(value="⚠️  Flag", interactive=False , variant="secondary")
                #stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False ,variant="secondary")
            with gr.Column(scale=3):
                clear_btn = gr.Button(value="🗑️  Clear", interactive=False , variant="stop")
    
            with gr.Accordion("Examples", open=True) as Examples_row:
                gr.Examples(examples=[
                [f"Tell me about the latest news in the world ?"],
                [f"Tell me about the increase in the price of Bitcoin ?"],
                [f"Tell me about the actual situation in Ukraine ?"],
                [f"How true is the news about the increase in the price of oil ?"],
                [f"Tell me about current situation in palestinian ?"],
                [f"Tell me about the current situation in Afghanistan ?"],
                [f"what are the agenda of the United Nations ?"],
                ["how trump's compain going ?"],
            ],inputs=[textbox], label="Examples")


                
            with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(minimum=0.1, maximum=1.0, value=0.2, step=0.1, interactive=True, label="Temperature",)
                    top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P",)
                    max_output_tokens = gr.Slider(minimum=0, maximum=4096, value=1024, step=64, interactive=True, label="Max output tokens",)
        

#  ================================================================================================================================
    btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]

    upvote_btn.click(
        upvote_last_response,
        [],
        btn_list,
    )

    downvote_btn.click(
        downvote_last_response,
        [],
        btn_list
    )
    flag_btn.click(
        flag_last_response,
        [],
        btn_list,
    )
    regenerate_btn.click(
        remove_last_response,
        [chatbot],
        [textbox , chatbot] + btn_list,
    ).then(
        chat,
        [ chatbot, textbox, max_output_tokens, temperature, top_p],
        [textbox, chatbot] + btn_list
    )

    clear_btn.click(
        clear_chat,
        [chatbot],
        [textbox , chatbot] + btn_list,
    )

    submit_btn.click(    
        chat ,
        [ chatbot, textbox , max_output_tokens, temperature, top_p],
        [textbox ,chatbot] + btn_list , 
                )
 
 #  ================================================================================================================================
demo.launch()



#  ================================================================================================================================
