import open_clip
import torch

clip_tokenizer = open_clip.get_tokenizer(model_name="ViT-g-14", context_length=300)
caption = "A man wearing a red/orange shirt and grey sneakers stands on the sidewalk next to a tall green clock. A parked bike is chained far away while a white car with its headlight off is next to a parking meter on the brick sidewalk. Nearby, there is a sign, a tall brick building, varied trees, sparse trees, and lamp posts. The scene also includes a white work truck parked on a clean street and a van parked nearby a grey wall."
tokens = clip_tokenizer(caption, context_length=300)
print("tokens", tokens)
n_tokens = torch.sum(tokens!=0)
n_words = len(caption.split(' '))
print("n_tokens", n_tokens)
print("n_words", n_words)
print("ratio:", n_words / n_tokens)