from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOllama
import base64
from io import BytesIO
from langchain_core.output_parsers import StrOutputParser
from PIL import Image

base_url = "https://7f8c-34-132-9-247.ngrok-free.app"
llm = ChatOllama(model="bakllava", temperature=0,base_url=base_url)


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def prompt_func(data):
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

image_path = './Pie_chart.jpeg'
pil_image = Image.open(image_path)

image_b64 = convert_to_base64(pil_image)
chain = prompt_func | llm | StrOutputParser()
query_chain = chain.invoke(
    {"text": "tell me percentage of visit friends or relatives", "image": convert_to_base64(pil_image)}
)

print(query_chain)