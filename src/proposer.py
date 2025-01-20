import random

import torch
from openai import OpenAI
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


class Proposer:
    def __init__(
        self,
        caption_model_name="Salesforce/blip2-opt-2.7b",
        llm_model_name="gpt-4o",
        subset_size=20,
    ):
        self.processor = Blip2Processor.from_pretrained(caption_model_name)
        self.model = (
            Blip2ForConditionalGeneration.from_pretrained(caption_model_name).to("cuda")
            if torch.cuda.is_available()
            else Blip2ForConditionalGeneration.from_pretrained(caption_model_name)
        )
        self.openai_client = OpenAI()
        self.llm_model_name = llm_model_name
        self.subset_size = subset_size

    def _generate_captions(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(
            self.model.device
        )
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return generated_text

    def propose(self, image_set_a: list[Image.Image], image_set_b: list[Image.Image]):
        subset_a = random.sample(image_set_a, min(len(image_set_a), self.subset_size))
        subset_b = random.sample(image_set_b, min(len(image_set_b), self.subset_size))

        captions_a = self._generate_captions(subset_a)
        captions_b = self._generate_captions(subset_b)

        prompt = f"""\
The following are the result of captioning two groups of images:
Group A: {captions_a}
Group B: {captions_b}
I am trying to figure out the major differences between these two groups so I can better understand my data.
Come up with 10 distinct concepts that are more likely to be true for Group A compared to Group B. Please write a list of captions (separated by bullet points "*"). For example:
* "a dog next to a horse"
* "a car in the rain"
Do not talk about the caption, e.g., "caption with one word" and do not list more than one concept. The hypothesis should be a caption, so hypotheses like "more of ...", "presence of ...", "images with ..." are incorrect. Also do not enumerate possibilities within parentheses. Here are examples of bad outputs and their corrections:\n* INCORRECT: "various nature environments like lakes, forests, and mountains" CORRECTED: "nature"
* INCORRECT: "images of household object (e.g. bowl, vacuum, lamp)" CORRECTED: "household objects"
* INCORRECT: "Presence of baby animals" CORRECTED: "baby animals"
* INCORRECT: "Different types of vehicles including cars, trucks, boats, and RVs" CORRECTED: "vehicles"
* INCORRECT: "Images involving interaction between humans and animals" CORRECTED: "interaction between humans and animals"
* INCORRECT: "More realistic images" CORRECTED: "realistic images"
* INCORRECT: "Insects (cockroach, dragonfly, grasshopper)" CORRECTED: "insects"
Again, I want to figure out what kind of distribution shift are there. List properties that hold more often for the images (not captions) in group A compared to group B. Answer with a list (separated by bullet points "*").
"""
        completion = self.openai_client.chat.completions.create(
            model=self.llm_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data scientist working on a computer vision project. You have two sets of images and captions. You want to understand the differences between the two sets. You are asking the model to generate a list of 10 distinct concepts that are more likely to be true for Group A compared to Group B.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        differences = [
            choice.message.content
            for choice in completion.choices
            if choice.message.content is not None
        ]
        return differences
