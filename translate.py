from transformers import BartTokenizer, BartForConditionalGeneration


def text2emoji():
    path = "models/bart-base-emoji-epoch19"
    tokenizer = BartTokenizer.from_pretrained(path)
    generator = BartForConditionalGeneration.from_pretrained(path)

    # sentence = "I love the weather in Alaska!"
    # sentence = "good night!"
    # sentence = "I am angry!"
    sentence = "Happy Thanksgiving!"
    # sentence = "Merry Chrismas! This is the gift for you."
    # sentence = "I was bullied for 3 years by five boys. In my opinion, bullying is disgusting but it continues to be in life hoping to destroy people’s lives when in fact it makes them stronger."
    # sentence = "Being a nurse is a rollercoaster of emotions, from comforting patients to dealing with medical emergencies."

    inputs = tokenizer(sentence, return_tensors="pt")
    generated_ids = generator.generate(inputs["input_ids"], num_beams=6, do_sample=True, max_length=100)
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True).replace(" ", "")

    print(decoded)


if __name__ == "__main__":
    text2emoji()

