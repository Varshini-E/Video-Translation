import srt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile
import os


class SimpleTranslator:
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-de"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.temp_dir = tempfile.TemporaryDirectory(prefix="translator_")

    def translate_subs(self, src_path: str) -> str:
        with open(src_path, encoding="utf-8") as f:
            subs = list(srt.parse(f.read()))
        
        translated_subs = []
        for sub in subs:
            inputs = self.tokenizer(sub.content, return_tensors="pt", truncation=True)
            ids = self.model.generate(**inputs)
            text = self.tokenizer.decode(ids[0], skip_special_tokens=True)
            translated_subs.append(srt.Subtitle(index=sub.index, start=sub.start, end=sub.end, content=text))
        
        dest_path = os.path.join(self.temp_dir.name, "translated_subs.srt")
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(translated_subs))
        return dest_path
