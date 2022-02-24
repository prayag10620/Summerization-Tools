
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# initialize the model architecture and weights
model = T5ForConditionalGeneration.from_pretrained("t5-large")
# initialize the model tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-large")


def Abstractive_Summarizer(article):
    ln = len(article)
    print(ln)
    mxl = (ln*50)//100
    req_mxl = (mxl*50)//100
    req_mnl = (mxl*15)//100
    article = article.strip().replace("\n","")
    # encode the text into tensor of integers using the appropriate tokenizer
    inputs = tokenizer.encode("summarize: " + article, return_tensors="pt", max_length=mxl, truncation=True)
    # generate the summarization output
    outputs = model.generate(
        inputs, 
        max_length=req_mxl, 
        min_length=req_mnl, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True)
    # just for debugging
    #print(outputs)
    req_summary = tokenizer.decode(outputs[0])
    return str(req_summary)
#artical = """For example, when you’re planning your next trip, Discover might show an article with the best places to eat or sights to see. Suddenly, a travel article published three months ago is timely for you. This can also be useful as you’re taking up a new hobby or going deeper on a long-time interest. Using the Topic Layer in the Knowledge Graph, Discover can predict your level of expertise on a topic and help you further develop those interests. If you’re learning to play guitar, for example, you might see beginner content about learning chords. If you’re already a skilled musician, you may see a video on more advanced techniques.Discover is unique because its one step ahead: it helps you come across the things you haven't even started looking for."""
#print(Abstractive_Summarizer(artical))
#def Abstractive_Summarizer(article):
#    return str(article)