
import torch
from PIL import Image
from gtts import gTTS
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import matplotlib.pyplot as plt
from urllib.request import urlretrieve


def predict_image_caption(image_path):
    
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = 32
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    images = []
    image_path = r'/Users/srisathvig/Documents/Projects/1 - VIEW ASSIST AI/Image Text Recognization/sample image.jpeg'
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


# mahesh request: Add a module which can convert text to speech 

def caption_to_speech(text):
    """
    this module helps to convert text to speech via GTTS lib

    """
    tts = gTTS(text=text)
    filename = "voice.mp3"
    tts.save(filename)
    # playsound.playsound(filename)
    print(filename)

#demo of code 
if __name__ == "__main__":

    image_path = r'/Users/srisathvig/Documents/Projects/1 - VIEW ASSIST AI/Image Text Recognization/sample image.jpeg'

    img_array =Image.open(image_path).convert('RGB')
    plt.imshow(img_array);
    text = predict_image_caption('sample image.jpeg')
    print(text)
    speech_to_text = " ".join(text)
    caption_to_speech(speech_to_text)



