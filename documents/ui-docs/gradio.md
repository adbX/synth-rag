# How to Create a Chatbot with Gradio (https://www.gradio.app/guides/creating-a-chatbot-fast)

## Introduction

Chatbots are a popular application of large language models (LLMs). Using Gradio, you can easily build a chat application and share that with your users, or try it yourself using an intuitive UI.

This tutorial uses `gr.ChatInterface()`, which is a high-level abstraction that allows you to create your chatbot UI fast, often with a _few lines of Python_. It can be easily adapted to support multimodal chatbots, or chatbots that require further customization.

**Prerequisites**: please make sure you are using the latest version of Gradio:

```bash
$ pip install --upgrade gradio
```

## Note for OpenAI-API compatible endpoints

If you have a chat server serving an OpenAI-API compatible endpoint (such as Ollama), you can spin up a ChatInterface in a single line of Python. First, also run `pip install openai`. Then, with your own URL, model, and optional token:

```python
import gradio as gr

gr.load_chat("http://localhost:11434/v1/", model="llama3.2", token="***").launch()
```

Read about `gr.load_chat` in [the docs](https://www.gradio.app/docs/gradio/load_chat). If you have your own model, keep reading to see how to create an application around any chat model in Python!

## Defining a chat function

To create a chat application with `gr.ChatInterface()`, the first thing you should do is define your **chat function**. In the simplest case, your chat function should accept two arguments: `message` and `history` (the arguments can be named anything, but must be in this order).

- `message`: a `str` representing the user's most recent message.
- `history`: a list of openai-style dictionaries with `role` and `content` keys, representing the previous conversation history. May also include additional keys representing message metadata.

The `history` would look like this:

```python
[
    {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Paris"}]}
]
```

while the next `message` would be:

```py
"And what is its largest city?"
```

Your chat function simply needs to return: 

* a `str` value, which is the chatbot's response based on the chat `history` and most recent `message`, for example, in this case:

```
Paris is also the largest city.
```

Let's take a look at a few example chat functions:

**Example: a chatbot that randomly responds with yes or no**

Let's write a chat function that responds `Yes` or `No` randomly.

Here's our chat function:

```python
import random

def random_response(message, history):
    return random.choice(["Yes", "No"])
```

Now, we can plug this into `gr.ChatInterface()` and call the `.launch()` method to create the web interface:

```python
import gradio as gr

gr.ChatInterface(
    fn=random_response, 
).launch()
```

That's it! Here's our running demo, try it out:

<gradio-app space='gradio/chatinterface_random_response'></gradio-app>

**Example: a chatbot that alternates between agreeing and disagreeing**

Of course, the previous example was very simplistic, it didn't take user input or the previous history into account! Here's another simple example showing how to incorporate a user's input as well as the history.

```python
import gradio as gr

def alternatingly_agree(message, history):
    if len([h for h in history if h['role'] == "assistant"]) % 2 == 0:
        return f"Yes, I do think that: {message}"
    else:
        return "I don't think so"

gr.ChatInterface(
    fn=alternatingly_agree, 
).launch()
```

We'll look at more realistic examples of chat functions in our next Guide, which shows [examples of using `gr.ChatInterface` with popular LLMs](../guides/chatinterface-examples). 

## Streaming chatbots

In your chat function, you can use `yield` to generate a sequence of partial responses, each replacing the previous ones. This way, you'll end up with a streaming chatbot. It's that simple!

```python
import time
import gradio as gr

def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.3)
        yield "You typed: " + message[: i+1]

gr.ChatInterface(
    fn=slow_echo, 
).launch()
```

While the response is streaming, the "Submit" button turns into a "Stop" button that can be used to stop the generator function.
            <div class='tip'>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/>
                    <path d="M9 18h6"/>
                    <path d="M10 22h4"/>
                </svg>
                <div><p>Even though you are yielding the latest message at each iteration, Gradio only sends the "diff" of each message from the server to the frontend, which reduces latency and data consumption over your network.</p></div>
            </div>
                

## Customizing the Chat UI

If you're familiar with Gradio's `gr.Interface` class, the `gr.ChatInterface` includes many of the same arguments that you can use to customize the look and feel of your Chatbot. For example, you can:

- add a title and description above your chatbot using `title` and `description` arguments.
- add a theme or custom css using `theme` and `css` arguments respectively in the `launch()` method.
- add `examples` and even enable `cache_examples`, which make your Chatbot easier for users to try it out.
- customize the chatbot (e.g. to change the height or add a placeholder) or textbox (e.g. to add a max number of characters or add a placeholder).

**Adding examples**

You can add preset examples to your `gr.ChatInterface` with the `examples` parameter, which takes a list of string examples. Any examples will appear as "buttons" within the Chatbot before any messages are sent. If you'd like to include images or other files as part of your examples, you can do so by using this dictionary format for each example instead of a string: `{"text": "What's in this image?", "files": ["cheetah.jpg"]}`. Each file will be a separate message that is added to your Chatbot history.

You can change the displayed text for each example by using the `example_labels` argument. You can add icons to each example as well using the `example_icons` argument. Both of these arguments take a list of strings, which should be the same length as the `examples` list.

If you'd like to cache the examples so that they are pre-computed and the results appear instantly, set `cache_examples=True`.

**Customizing the chatbot or textbox component**

If you want to customize the `gr.Chatbot` or `gr.Textbox` that compose the `ChatInterface`, then you can pass in your own chatbot or textbox components. Here's an example of how we to apply the parameters we've discussed in this section:

```python
import gradio as gr

def yes_man(message, history):
    if message.endswith("?"):
        return "Yes"
    else:
        return "Ask me anything!"

gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
    title="Yes Man",
    description="Ask Yes Man any question",
    examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
    cache_examples=True,
).launch(theme="ocean")
```

Here's another example that adds a "placeholder" for your chat interface, which appears before the user has started chatting. The `placeholder` argument of `gr.Chatbot` accepts Markdown or HTML:

```python
gr.ChatInterface(
    yes_man,
    chatbot=gr.Chatbot(placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything"),
...
```

The placeholder appears vertically and horizontally centered in the chatbot.

## Multimodal Chat Interface

You may want to add multimodal capabilities to your chat interface. For example, you may want users to be able to upload images or files to your chatbot and ask questions about them. You can make your chatbot "multimodal" by passing in a single parameter (`multimodal=True`) to the `gr.ChatInterface` class.

When `multimodal=True`, the signature of your chat function changes slightly: the first parameter of your function (what we referred to as `message` above) should accept a dictionary consisting of the submitted text and uploaded files that looks like this: 

```py
{
    "text": "user input", 
    "files": [
        "updated_file_1_path.ext",
        "updated_file_2_path.ext", 
        ...
    ]
}
```

This second parameter of your chat function, `history`, will be in the same openai-style dictionary format as before. However, if the history contains uploaded files, the `content` key will be a dictionary with a "type" key whose value is "file" and the file will be represented as a dictionary. All the files will be grouped in message in the history. So after uploading two files and asking a question, your history might look like this:

```python
[
    {"role": "user", "content": [{"type": "file", "file": {"path": "cat1.png"}},
                                 {"type": "file", "file": {"path": "cat1.png"}},
                                 {"type": "text", "text": "What's the difference between these two images?"}]}
]
```

The return type of your chat function does *not change* when setting `multimodal=True` (i.e. in the simplest case, you should still return a string value). We discuss more complex cases, e.g. returning files [below](#returning-complex-responses).

If you are customizing a multimodal chat interface, you should pass in an instance of `gr.MultimodalTextbox` to the `textbox` parameter. You can customize the `MultimodalTextbox` further by passing in the `sources` parameter, which is a list of sources to enable. Here's an example that illustrates how to set up and customize and multimodal chat interface:
 

```python
import gradio as gr

def count_images(message, history):
    num_images = len(message["files"])
    total_images = 0
    for message in history:
        for content in message["content"]:
            if content["type"] == "file":
                total_images += 1
    return f"You just uploaded {num_images} images, total uploaded: {total_images+num_images}"

demo = gr.ChatInterface(
    fn=count_images, 
    examples=[
        {"text": "No files", "files": []}
    ], 
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple", file_types=["image"], sources=["upload", "microphone"])
)

demo.launch()
```

## Additional Inputs

You may want to add additional inputs to your chat function and expose them to your users through the chat UI. For example, you could add a textbox for a system prompt, or a slider that sets the number of tokens in the chatbot's response. The `gr.ChatInterface` class supports an `additional_inputs` parameter which can be used to add additional input components.

The `additional_inputs` parameters accepts a component or a list of components. You can pass the component instances directly, or use their string shortcuts (e.g. `"textbox"` instead of `gr.Textbox()`). If you pass in component instances, and they have _not_ already been rendered, then the components will appear underneath the chatbot within a `gr.Accordion()`. 

Here's a complete example:

```python
import gradio as gr
import time

def echo(message, history, system_prompt, tokens):
    response = f"System prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i + 1]

demo = gr.ChatInterface(
    echo,
    additional_inputs=[
        gr.Textbox("You are helpful AI.", label="System Prompt"),
        gr.Slider(10, 100),
    ],
)

demo.launch()

```

If the components you pass into the `additional_inputs` have already been rendered in a parent `gr.Blocks()`, then they will _not_ be re-rendered in the accordion. This provides flexibility in deciding where to lay out the input components. In the example below, we position the `gr.Textbox()` on top of the Chatbot UI, while keeping the slider underneath.

```python
import gradio as gr
import time

def echo(message, history, system_prompt, tokens):
    response = f"System prompt: {system_prompt}\n Message: {message}."
    for i in range(min(len(response), int(tokens))):
        time.sleep(0.05)
        yield response[: i+1]

with gr.Blocks() as demo:
    system_prompt = gr.Textbox("You are helpful AI.", label="System Prompt")
    slider = gr.Slider(10, 100, render=False)

    gr.ChatInterface(
        echo, additional_inputs=[system_prompt, slider],
    )

demo.launch()
```

**Examples with additional inputs**

You can also add example values for your additional inputs. Pass in a list of lists to the `examples` parameter, where each inner list represents one sample, and each inner list should be `1 + len(additional_inputs)` long. The first element in the inner list should be the example value for the chat message, and each subsequent element should be an example value for one of the additional inputs, in order. When additional inputs are provided, examples are rendered in a table underneath the chat interface.

If you need to create something even more custom, then its best to construct the chatbot UI using the low-level `gr.Blocks()` API. We have [a dedicated guide for that here](/guides/creating-a-custom-chatbot-with-blocks).

## Additional Outputs

In the same way that you can accept additional inputs into your chat function, you can also return additional outputs. Simply pass in a list of components to the `additional_outputs` parameter in `gr.ChatInterface` and return additional values for each component from your chat function. Here's an example that extracts code and outputs it into a separate `gr.Code` component:

```python
import gradio as gr

python_code = """
def fib(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)
"""

js_code = """
function fib(n) {
    if (n <= 0) return 0;
    if (n === 1) return 1;
    return fib(n - 1) + fib(n - 2);
}
"""

def chat(message, history):
    if "python" in message.lower():
        return "Type Python or JavaScript to see the code.", gr.Code(language="python", value=python_code)
    elif "javascript" in message.lower():
        return "Type Python or JavaScript to see the code.", gr.Code(language="javascript", value=js_code)
    else:
        return "Please ask about Python or JavaScript.", None

with gr.Blocks() as demo:
    code = gr.Code(render=False)
    with gr.Row():
        with gr.Column():
            gr.Markdown("<center><h1>Write Python or JavaScript</h1></center>")
            gr.ChatInterface(
                chat,
                examples=["Python", "JavaScript"],
                additional_outputs=[code],
                api_name="chat",
            )
        with gr.Column():
            gr.Markdown("<center><h1>Code Artifacts</h1></center>")
            code.render()

demo.launch()

```

**Note:** unlike the case of additional inputs, the components passed in `additional_outputs` must be already defined in your `gr.Blocks` context -- they are not rendered automatically. If you need to render them after your `gr.ChatInterface`, you can set `render=False` when they are first defined and then `.render()` them in the appropriate section of your `gr.Blocks()` as we do in the example above.

## Returning Complex Responses

We mentioned earlier that in the simplest case, your chat function should return a `str` response, which will be rendered as Markdown in the chatbot. However, you can also return more complex responses as we discuss below:


**Returning files or Gradio components**

Currently, the following Gradio components can be displayed inside the chat interface:
* `gr.Image`
* `gr.Plot`
* `gr.Audio`
* `gr.HTML`
* `gr.Video`
* `gr.Gallery`
* `gr.File`

Simply return one of these components from your function to use it with `gr.ChatInterface`. Here's an example that returns an audio file:

```py
import gradio as gr

def music(message, history):
    if message.strip():
        return gr.Audio("https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav")
    else:
        return "Please provide the name of an artist"

gr.ChatInterface(
    music,
    textbox=gr.Textbox(placeholder="Which artist's music do you want to listen to?", scale=7),
).launch()
```

Similarly, you could return image files with `gr.Image`, video files with `gr.Video`, or arbitrary files with the `gr.File` component.

**Returning Multiple Messages**

You can return multiple assistant messages from your chat function simply by returning a `list` of messages, each of which is a valid chat type. This lets you, for example, send a message along with files, as in the following example:

```python
import gradio as gr

def echo_multimodal(message, history):
    response = []
    response.append("You wrote: '" + message["text"] + "' and uploaded:")
    if message.get("files"):
        for file in message["files"]:
            response.append(gr.File(value=file))
    return response

demo = gr.ChatInterface(
    echo_multimodal,
    multimodal=True,
    textbox=gr.MultimodalTextbox(file_count="multiple"),
    api_name="chat",
)

demo.launch()

```


**Displaying intermediate thoughts or tool usage**

The `gr.ChatInterface` class supports displaying intermediate thoughts or tool usage direct in the chatbot.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/gradio-guides/nested-thought.png)

 To do this, you will need to return a `gr.ChatMessage` object from your chat function. Here is the schema of the `gr.ChatMessage` data class as well as two internal typed dictionaries:
 
 ```py
MessageContent = Union[str, FileDataDict, FileData, Component]

@dataclass
class ChatMessage:
    content: MessageContent | list[MessageContent]
    metadata: MetadataDict = None
    options: list[OptionDict] = None

class MetadataDict(TypedDict):
    title: NotRequired[str]
    id: NotRequired[int | str]
    parent_id: NotRequired[int | str]
    log: NotRequired[str]
    duration: NotRequired[float]
    status: NotRequired[Literal["pending", "done"]]

class OptionDict(TypedDict):
    label: NotRequired[str]
    value: str
 ```
 
As you can see, the `gr.ChatMessage` dataclass is similar to the openai-style message format, e.g. it has a "content" key that refers to the chat message content. But it also includes a "metadata" key whose value is a dictionary. If this dictionary includes a "title" key, the resulting message is displayed as an intermediate thought with the title being displayed on top of the thought. Here's an example showing the usage:

```python
import gradio as gr
from gradio import ChatMessage
import time

sleep_time = 0.5

def simulate_thinking_chat(message, history):
    start_time = time.time()
    response = ChatMessage(
        content="",
        metadata={"title": "_Thinking_ step-by-step", "id": 0, "status": "pending"}
    )
    yield response

    thoughts = [
        "First, I need to understand the core aspects of the query...",
        "Now, considering the broader context and implications...",
        "Analyzing potential approaches to formulate a comprehensive answer...",
        "Finally, structuring the response for clarity and completeness..."
    ]

    accumulated_thoughts = ""
    for thought in thoughts:
        time.sleep(sleep_time)
        accumulated_thoughts += f"- {thought}\n\n"
        response.content = accumulated_thoughts.strip()
        yield response

    response.metadata["status"] = "done"
    response.metadata["duration"] = time.time() - start_time
    yield response

    response = [
        response,
        ChatMessage(
            content="Based on my thoughts and analysis above, my response is: This dummy repro shows how thoughts of a thinking LLM can be progressively shown before providing its final answer."
        )
    ]
    yield response


demo = gr.ChatInterface(
    simulate_thinking_chat,
    title="Thinking LLM Chat Interface 🤔",
)

demo.launch()

```

You can even show nested thoughts, which is useful for agent demos in which one tool may call other tools. To display nested thoughts, include "id" and "parent_id" keys in the "metadata" dictionary. Read our [dedicated guide on displaying intermediate thoughts and tool usage](/guides/agents-and-tool-usage) for more realistic examples.

**Providing preset responses**

When returning an assistant message, you may want to provide preset options that a user can choose in response. To do this, again, you will again return a `gr.ChatMessage` instance from your chat function. This time, make sure to set the `options` key specifying the preset responses.

As shown in the schema for `gr.ChatMessage` above, the value corresponding to the `options` key should be a list of dictionaries, each with a `value` (a string that is the value that should be sent to the chat function when this response is clicked) and an optional `label` (if provided, is the text displayed as the preset response instead of the `value`). 

This example illustrates how to use preset responses:

```python
import gradio as gr
import random

example_code = """
Here's an example Python lambda function:

lambda x: x + {}

Is this correct?
"""

def chat(message, history):
    if message == "Yes, that's correct.":
        return "Great!"
    else:
        return gr.ChatMessage(
            content=example_code.format(random.randint(1, 100)),
            options=[
                {"value": "Yes, that's correct.", "label": "Yes"},
                {"value": "No"}
            ]
        )

demo = gr.ChatInterface(
    chat,
    examples=["Write an example Python lambda function."],
    api_name="chat",
)

demo.launch()

```

## Modifying the Chatbot Value Directly

You may wish to modify the value of the chatbot with your own events, other than those prebuilt in the `gr.ChatInterface`. For example, you could create a dropdown that prefills the chat history with certain conversations or add a separate button to clear the conversation history. The `gr.ChatInterface` supports these events, but you need to use the `gr.ChatInterface.chatbot_value` as the input or output component in such events. In this example, we use a `gr.Radio` component to prefill the the chatbot with certain conversations:

```python
import gradio as gr
import random

def prefill_chatbot(choice):
    if choice == "Greeting":
        return [
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
    elif choice == "Complaint":
        return [
            {"role": "user", "content": "I'm not happy with the service."},
            {"role": "assistant", "content": "I'm sorry to hear that. Can you please tell me more about the issue?"}
        ]
    else:
        return []

def random_response(message, history):
    return random.choice(["Yes", "No"])

with gr.Blocks() as demo:
    radio = gr.Radio(["Greeting", "Complaint", "Blank"])
    chat = gr.ChatInterface(random_response, api_name="chat")
    radio.change(prefill_chatbot, radio, chat.chatbot_value)

demo.launch()

```

## Using Your Chatbot via API

Once you've built your Gradio chat interface and are hosting it on [Hugging Face Spaces](https://hf.space) or somewhere else, then you can query it with a simple API. The API route will be the name of the function you pass to the ChatInterface. So if `gr.ChatInterface(respond)`, then the API route is `/respond`. The endpoint just expects the user's message and will return the response, internally keeping track of the message history.

![](https://github.com/gradio-app/gradio/assets/1778297/7b10d6db-6476-4e2e-bebd-ecda802c3b8f)

To use the endpoint, you should use either the [Gradio Python Client](/guides/getting-started-with-the-python-client) or the [Gradio JS client](/guides/getting-started-with-the-js-client). Or, you can deploy your Chat Interface to other platforms, such as a:

* Slack bot [[tutorial]](../guides/creating-a-slack-bot-from-a-gradio-app)
* Website widget [[tutorial]](../guides/creating-a-website-widget-from-a-gradio-chatbot)

## Chat History

You can enable persistent chat history for your ChatInterface, allowing users to maintain multiple conversations and easily switch between them. When enabled, conversations are stored locally and privately in the user's browser using local storage. So if you deploy a ChatInterface e.g. on [Hugging Face Spaces](https://hf.space), each user will have their own separate chat history that won't interfere with other users' conversations. This means multiple users can interact with the same ChatInterface simultaneously while maintaining their own private conversation histories.

To enable this feature, simply set `gr.ChatInterface(save_history=True)` (as shown in the example in the next section). Users will then see their previous conversations in a side panel and can continue any previous chat or start a new one.

## Collecting User Feedback

To gather feedback on your chat model, set `gr.ChatInterface(flagging_mode="manual")` and users will be able to thumbs-up or thumbs-down assistant responses. Each flagged response, along with the entire chat history, will get saved in a CSV file in the app working directory (this can be configured via the `flagging_dir` parameter). 

You can also change the feedback options via `flagging_options` parameter. The default options are "Like" and "Dislike", which appear as the thumbs-up and thumbs-down icons. Any other options appear under a dedicated flag icon. This example shows a ChatInterface that has both chat history (mentioned in the previous section) and user feedback enabled:

```python
import time
import gradio as gr

def slow_echo(message, history):
    for i in range(len(message)):
        time.sleep(0.05)
        yield "You typed: " + message[: i + 1]

demo = gr.ChatInterface(
    slow_echo,
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True,
)

demo.launch()

```

Note that in this example, we set several flagging options: "Like", "Spam", "Inappropriate", "Other". Because the case-sensitive string "Like" is one of the flagging options, the user will see a thumbs-up icon next to each assistant message. The three other flagging options will appear in a dropdown under the flag icon.

## What's Next?

Now that you've learned about the `gr.ChatInterface` class and how it can be used to create chatbot UIs quickly, we recommend reading one of the following:

* [Our next Guide](../guides/chatinterface-examples) shows examples of how to use `gr.ChatInterface` with popular LLM libraries.
* If you'd like to build very custom chat applications from scratch, you can build them using the low-level Blocks API, as [discussed in this Guide](../guides/creating-a-custom-chatbot-with-blocks).
* Once you've deployed your Gradio Chat Interface, its easy to use in other applications because of the built-in API. Here's a tutorial on [how to deploy a Gradio chat interface as a Discord bot](../guides/creating-a-discord-bot-from-a-gradio-app).


# Using Popular LLM libraries and APIs (https://www.gradio.app/guides/chatinterface-examples)


In this Guide, we go through several examples of how to use `gr.ChatInterface` with popular LLM libraries and API providers.

We will cover the following libraries and API providers:

* [Llama Index](#llama-index)
* [LangChain](#lang-chain)
* [OpenAI](#open-ai)
* [Hugging Face `transformers`](#hugging-face-transformers)
* [SambaNova](#samba-nova)
* [Hyperbolic](#hyperbolic)
* [Anthropic's Claude](#anthropics-claude)

For many LLM libraries and providers, there exist community-maintained integration libraries that make it even easier to spin up Gradio apps. We reference these libraries in the appropriate sections below.

## Llama Index

Let's start by using `llama-index` on top of `openai` to build a RAG chatbot on any text or PDF files that you can demo and share in less than 30 lines of code. You'll need to have an OpenAI key for this example (keep reading for the free, open-source equivalent!)

```python
# This is a simple RAG chatbot built on top of Llama Index and Gradio. It allows you to upload any text or PDF files and ask questions about them!
# Before running this, make sure you have exported your OpenAI API key as an environment variable:
# export OPENAI_API_KEY="your-openai-api-key"

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader  
import gradio as gr

def answer(message, history):
    files = []
    for msg in history:
        if msg['role'] == "user" and isinstance(msg['content'], tuple):
            files.append(msg['content'][0])
    for file in message["files"]:
        files.append(file)

    documents = SimpleDirectoryReader(input_files=files).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return str(query_engine.query(message["text"]))

demo = gr.ChatInterface(
    answer,
    title="Llama Index RAG Chatbot",
    description="Upload any text or pdf files and ask questions about them!",
    textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt"]),
    multimodal=True,
    api_name="chat",
)

demo.launch()

```

## LangChain

Here's an example using `langchain` on top of `openai` to build a general-purpose chatbot. As before, you'll need to have an OpenAI key for this example.

```python
# This is a simple general-purpose chatbot built on top of LangChain and Gradio.
# Before running this, make sure you have exported your OpenAI API key as an environment variable:
# export OPENAI_API_KEY="your-openai-api-key"

from langchain_openai import ChatOpenAI  
from langchain.schema import AIMessage, HumanMessage  
import gradio as gr

model = ChatOpenAI(model="gpt-4o-mini")

def predict(message, history):
    history_langchain_format = []
    for msg in history:
        if msg['role'] == "user":
            history_langchain_format.append(HumanMessage(content=msg['content']))
        elif msg['role'] == "assistant":
            history_langchain_format.append(AIMessage(content=msg['content']))
    history_langchain_format.append(HumanMessage(content=message))
    gpt_response = model.invoke(history_langchain_format)
    return gpt_response.content

demo = gr.ChatInterface(
    predict,
    api_name="chat",
)

demo.launch()

```
            <div class='tip'>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/>
                    <path d="M9 18h6"/>
                    <path d="M10 22h4"/>
                </svg>
                <div><p>For quick prototyping, the community-maintained <a href='https://github.com/AK391/langchain-gradio'>langchain-gradio repo</a>  makes it even easier to build chatbots on top of LangChain.</p></div>
            </div>
                

## OpenAI

Of course, we could also use the `openai` library directy. Here a similar example to the LangChain , but this time with streaming as well:
            <div class='tip'>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/>
                    <path d="M9 18h6"/>
                    <path d="M10 22h4"/>
                </svg>
                <div><p>For quick prototyping, the  <a href='https://github.com/gradio-app/openai-gradio'>openai-gradio library</a> makes it even easier to build chatbots on top of OpenAI models.</p></div>
            </div>
                


## Hugging Face `transformers`

Of course, in many cases you want to run a chatbot locally. Here's the equivalent example using the SmolLM2-135M-Instruct model using the Hugging Face `transformers` library.

```python

from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cpu"  # "cuda" or "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

def predict(message, history):
    history.append({"role": "user", "content": message})
    input_text = tokenizer.apply_chat_template(history, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)  
    outputs = model.generate(inputs, max_new_tokens=100, temperature=0.2, top_p=0.9, do_sample=True)
    decoded = tokenizer.decode(outputs[0])
    response = decoded.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0]
    return response

demo = gr.ChatInterface(predict, api_name="chat")

demo.launch()

```

## SambaNova

The SambaNova Cloud API provides access to full-precision open-source models, such as the Llama family. Here's an example of how to build a Gradio app around the SambaNova API

```python
# This is a simple general-purpose chatbot built on top of SambaNova API. 
# Before running this, make sure you have exported your SambaNova API key as an environment variable:
# export SAMBANOVA_API_KEY="your-sambanova-api-key"

import os
import gradio as gr
from openai import OpenAI

api_key = os.getenv("SAMBANOVA_API_KEY")

client = OpenAI(
    base_url="https://api.sambanova.ai/v1/",
    api_key=api_key,
)

def predict(message, history):
    history.append({"role": "user", "content": message})
    stream = client.chat.completions.create(messages=history, model="Meta-Llama-3.1-70B-Instruct-8k", stream=True)
    chunks = []
    for chunk in stream:
        chunks.append(chunk.choices[0].delta.content or "")
        yield "".join(chunks)

demo = gr.ChatInterface(predict, api_name="chat")

demo.launch()


```
            <div class='tip'>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/>
                    <path d="M9 18h6"/>
                    <path d="M10 22h4"/>
                </svg>
                <div><p>For quick prototyping, the  <a href='https://github.com/gradio-app/sambanova-gradio'>sambanova-gradio library</a> makes it even easier to build chatbots on top of SambaNova models.</p></div>
            </div>
                

## Hyperbolic

The Hyperbolic AI API provides access to many open-source models, such as the Llama family. Here's an example of how to build a Gradio app around the Hyperbolic

```python
# This is a simple general-purpose chatbot built on top of Hyperbolic API. 
# Before running this, make sure you have exported your Hyperbolic API key as an environment variable:
# export HYPERBOLIC_API_KEY="your-hyperbolic-api-key"

import os
import gradio as gr
from openai import OpenAI

api_key = os.getenv("HYPERBOLIC_API_KEY")

client = OpenAI(
    base_url="https://api.hyperbolic.xyz/v1/",
    api_key=api_key,
)

def predict(message, history):
    history.append({"role": "user", "content": message})
    stream = client.chat.completions.create(messages=history, model="gpt-4o-mini", stream=True)
    chunks = []
    for chunk in stream:
        chunks.append(chunk.choices[0].delta.content or "")
        yield "".join(chunks)

demo = gr.ChatInterface(predict, api_name="chat")

demo.launch()


```
            <div class='tip'>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/>
                    <path d="M9 18h6"/>
                    <path d="M10 22h4"/>
                </svg>
                <div><p>For quick prototyping, the  <a href='https://github.com/HyperbolicLabs/hyperbolic-gradio'>hyperbolic-gradio library</a> makes it even easier to build chatbots on top of Hyperbolic models.</p></div>
            </div>
                


## Anthropic's Claude 

Anthropic's Claude model can also be used via API. Here's a simple 20 questions-style game built on top of the Anthropic API:

```python
# This is a simple 20 questions-style game built on top of the Anthropic API.
# Before running this, make sure you have exported your Anthropic API key as an environment variable:
# export ANTHROPIC_API_KEY="your-anthropic-api-key"

import anthropic  
import gradio as gr

client = anthropic.Anthropic()

def predict(message, history):
    keys_to_keep = ["role", "content"]
    history = [{k: d[k] for k in keys_to_keep if k in d} for d in history]
    history.append({"role": "user", "content": message})
    if len(history) > 20:
        history.append({"role": "user", "content": "DONE"})
    output = client.messages.create(
        messages=history,  
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        system="You are guessing an object that the user is thinking of. You can ask 10 yes/no questions. Keep asking questions until the user says DONE"
    )
    return {
        "role": "assistant",
        "content": output.content[0].text,  
        "options": [{"value": "Yes"}, {"value": "No"}]
    }

placeholder = """
<center><h1>10 Questions</h1><br>Think of a person, place, or thing. I'll ask you 10 yes/no questions to try and guess it.
</center>
"""

demo = gr.ChatInterface(
    predict,
    examples=["Start!"],
    chatbot=gr.Chatbot(placeholder=placeholder),
    api_name="chat",
)

demo.launch()

```

# Building Conversational Chatbots with Gradio (https://www.gradio.app/guides/conversational-chatbot)


## Introduction

The next generation of AI user interfaces is moving towards audio-native experiences. Users will be able to speak to chatbots and receive spoken responses in return. Several models have been built under this paradigm, including GPT-4o and [mini omni](https://github.com/gpt-omni/mini-omni).

In this guide, we'll walk you through building your own conversational chat application using mini omni as an example. You can see a demo of the finished app below:

<video src="https://github.com/user-attachments/assets/db36f4db-7535-49f1-a2dd-bd36c487ebdf" controls
height="600" width="600" style="display: block; margin: auto;" autoplay="true" loop="true">
</video>

## Application Overview

Our application will enable the following user experience:

1. Users click a button to start recording their message
2. The app detects when the user has finished speaking and stops recording
3. The user's audio is passed to the omni model, which streams back a response
4. After omni mini finishes speaking, the user's microphone is reactivated
5. All previous spoken audio, from both the user and omni, is displayed in a chatbot component

Let's dive into the implementation details.

## Processing User Audio

We'll stream the user's audio from their microphone to the server and determine if the user has stopped speaking on each new chunk of audio.

Here's our `process_audio` function:

```python
import numpy as np
from utils import determine_pause

def process_audio(audio: tuple, state: AppState):
    if state.stream is None:
        state.stream = audio[1]
        state.sampling_rate = audio[0]
    else:
        state.stream = np.concatenate((state.stream, audio[1]))

    pause_detected = determine_pause(state.stream, state.sampling_rate, state)
    state.pause_detected = pause_detected

    if state.pause_detected and state.started_talking:
        return gr.Audio(recording=False), state
    return None, state
```

This function takes two inputs:
1. The current audio chunk (a tuple of `(sampling_rate, numpy array of audio)`)
2. The current application state

We'll use the following `AppState` dataclass to manage our application state:

```python
from dataclasses import dataclass

@dataclass
class AppState:
    stream: np.ndarray | None = None
    sampling_rate: int = 0
    pause_detected: bool = False
    stopped: bool = False
    conversation: list = []
```

The function concatenates new audio chunks to the existing stream and checks if the user has stopped speaking. If a pause is detected, it returns an update to stop recording. Otherwise, it returns `None` to indicate no changes.

The implementation of the `determine_pause` function is specific to the omni-mini project and can be found [here](https://huggingface.co/spaces/gradio/omni-mini/blob/eb027808c7bfe5179b46d9352e3fa1813a45f7c3/app.py#L98).

## Generating the Response

After processing the user's audio, we need to generate and stream the chatbot's response. Here's our `response` function:

```python
import io
import tempfile
from pydub import AudioSegment

def response(state: AppState):
    if not state.pause_detected and not state.started_talking:
        return None, AppState()
    
    audio_buffer = io.BytesIO()

    segment = AudioSegment(
        state.stream.tobytes(),
        frame_rate=state.sampling_rate,
        sample_width=state.stream.dtype.itemsize,
        channels=(1 if len(state.stream.shape) == 1 else state.stream.shape[1]),
    )
    segment.export(audio_buffer, format="wav")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_buffer.getvalue())
    
    state.conversation.append({"role": "user",
                                "content": {"path": f.name,
                                "mime_type": "audio/wav"}})
    
    output_buffer = b""

    for mp3_bytes in speaking(audio_buffer.getvalue()):
        output_buffer += mp3_bytes
        yield mp3_bytes, state

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(output_buffer)
    
    state.conversation.append({"role": "assistant",
                    "content": {"path": f.name,
                                "mime_type": "audio/mp3"}})
    yield None, AppState(conversation=state.conversation)
```

This function:
1. Converts the user's audio to a WAV file
2. Adds the user's message to the conversation history
3. Generates and streams the chatbot's response using the `speaking` function
4. Saves the chatbot's response as an MP3 file
5. Adds the chatbot's response to the conversation history

Note: The implementation of the `speaking` function is specific to the omni-mini project and can be found [here](https://huggingface.co/spaces/gradio/omni-mini/blob/main/app.py#L116).

## Building the Gradio App

Now let's put it all together using Gradio's Blocks API:

```python
import gradio as gr

def start_recording_user(state: AppState):
    if not state.stopped:
        return gr.Audio(recording=True)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_audio = gr.Audio(
                label="Input Audio", sources="microphone", type="numpy"
            )
        with gr.Column():
            chatbot = gr.Chatbot(label="Conversation")
            output_audio = gr.Audio(label="Output Audio", streaming=True, autoplay=True)
    state = gr.State(value=AppState())

    stream = input_audio.stream(
        process_audio,
        [input_audio, state],
        [input_audio, state],
        stream_every=0.5,
        time_limit=30,
    )
    respond = input_audio.stop_recording(
        response,
        [state],
        [output_audio, state]
    )
    respond.then(lambda s: s.conversation, [state], [chatbot])

    restart = output_audio.stop(
        start_recording_user,
        [state],
        [input_audio]
    )
    cancel = gr.Button("Stop Conversation", variant="stop")
    cancel.click(lambda: (AppState(stopped=True), gr.Audio(recording=False)), None,
                [state, input_audio], cancels=[respond, restart])

if __name__ == "__main__":
    demo.launch()
```

This setup creates a user interface with:
- An input audio component for recording user messages
- A chatbot component to display the conversation history
- An output audio component for the chatbot's responses
- A button to stop and reset the conversation

The app streams user audio in 0.5-second chunks, processes it, generates responses, and updates the conversation history accordingly.

## Conclusion

This guide demonstrates how to build a conversational chatbot application using Gradio and the mini omni model. You can adapt this framework to create various audio-based chatbot demos. To see the full application in action, visit the Hugging Face Spaces demo: https://huggingface.co/spaces/gradio/omni-mini

Feel free to experiment with different models, audio processing techniques, or user interface designs to create your own unique conversational AI experiences!
