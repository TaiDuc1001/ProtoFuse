from torch import nn
import torch
from tqdm import tqdm
import pickle
import json
import random
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from enum import Enum
from transformers import GPT2LMHeadModel
from typing import Tuple, Optional
import os
import numpy as np


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

    
class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        

class DeCap(nn.Module):
    def __init__(self, prefix_size: int = 512):
        super(DeCap, self).__init__()
        from transformers import GPT2Config
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=768,
            n_layer=4,
            n_head=4,
            n_inner=None,
            activation_function='gelu_new',
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-05,
            initializer_range=0.02,
            summary_type='cls_index',
            summary_use_proj=True,
            summary_activation=None,
            summary_first_dropout=0.1,
            summary_proj_to_labels=True,
            scale_attn_weights=True,
            use_cache=True,
            scale_attn_by_inverse_layer_idx=False,
            reorder_and_upcast_attn=False,
            bos_token_id=50256,
            pad_token_id=50256,
            eos_token_id=50256
        )
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, self.embedding_size))
        
    def forward(self, clip_features, tokens):
        embedding_text = self.decoder.transformer.wte(tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1, 1, self.embedding_size)
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out


class APTDecoder:
    def __init__(self, 
                 model_weights_path: str = './coco_prefix-009.pt',
                 coco_train_path: str = './coco_train.json',
                 device: str = 'cuda',
                 clip_model_name: str = "ViT-B/32"):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = _Tokenizer()
        
        self.clip_model, _ = clip.load(clip_model_name, device=self.device, jit=False)
        self.clip_tokenizer = clip.tokenize
        
        self.model = DeCap()
        if os.path.exists(model_weights_path):
            try:
                state_dict = torch.load(model_weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Could not load model weights: {e}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.text_features = None
        self.captions = []
        
        self._load_support_memory_optimized(coco_train_path)
    
    def _load_support_memory_optimized(self, coco_train_path: str):
        cache_path = coco_train_path.replace('.json', '_embeddings_cache.pt')
        
        if os.path.exists(cache_path):
            print("Loading cached COCO embeddings...")
            cache_data = torch.load(cache_path, map_location=self.device)
            self.text_features = cache_data['text_features'].to(self.device)
            self.captions = cache_data['captions']
            print(f"Loaded {len(self.captions)} cached embeddings")
            return
        
        with open(coco_train_path, 'r') as f:
            data = json.load(f)
        
        text_features = []
        batch_size = 2000
        
        self.clip_model.eval()
        
        for i in tqdm(range(0, len(data), batch_size), desc="Processing captions"):
            batch_texts = data[i:i+batch_size]
            with torch.no_grad():
                texts_token = self.clip_tokenizer(batch_texts).to(self.device)
                text_feature = self.clip_model.encode_text(texts_token)
                text_features.append(text_feature.cpu())
                self.captions.extend(batch_texts)

        self.text_features = torch.cat(text_features, dim=0)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True).float()
        
        torch.save({
            'text_features': self.text_features,
            'captions': self.captions
        }, cache_path)
        print(f"Cached {len(self.captions)} embeddings to {cache_path}")
        
        self.text_features = self.text_features.to(self.device)
    
    def _decode_single(self, prefix_embedding, entry_length: int = 30, temperature: float = 1.0):
        self.model.eval()
        embedding_cat = self.model.clip_project(prefix_embedding).reshape(1, 1, -1)
        tokens = None
        
        for i in range(entry_length):
            outputs = self.model.decoder(inputs_embeds=embedding_cat)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = self.model.decoder.transformer.wte(next_token)

            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            
            if next_token.item() == 49407:
                break
            
            embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
        
        try:
            output_list = list(tokens.squeeze().cpu().numpy()) # type: ignore
            output = self.tokenizer.decode(output_list)
        except:
            output = 'None'
        
        return output
    
    def _decode_batch(self, prefix_embeddings, entry_length: int = 30, temperature: float = 1.0):
        batch_size = prefix_embeddings.shape[0]
        prefix_projections = self.model.clip_project(prefix_embeddings)
        prefix_projections = prefix_projections.unsqueeze(1)
        bos_token = torch.full((batch_size, 1), 50256, dtype=torch.long, device=self.device)
        bos_embeddings = self.model.decoder.transformer.wte(bos_token)
        input_embeddings = torch.cat([prefix_projections, bos_embeddings], dim=1)
        attention_mask = torch.ones(batch_size, input_embeddings.shape[1], dtype=torch.long, device=self.device)
        generated = self.model.decoder.generate(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            max_length=entry_length + 2,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            eos_token_id=50256, 
            num_return_sequences=1,
            use_cache=True
        )
        generated_texts = []
        for i in range(batch_size):
            tokens = generated[i, 2:]
            eos_positions = (tokens == 50256).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                tokens = tokens[:eos_positions[0]]
            
            try:
                text = self.tokenizer.decode(tokens.cpu().tolist())
                text = text.replace('<|startoftext|>', '').replace('<|endoftext|>', '').strip()
                generated_texts.append(text)
            except:
                generated_texts.append('')
        
        return generated_texts
    
    def decode_from_apt_embedding_batch(self, learned_embeddings, entry_length: int = 30, temperature: float = 1.0):
        learned_embeddings = learned_embeddings.to(self.device)
        batch_size = learned_embeddings.shape[0]
        
        if self.text_features is not None:
            sim = learned_embeddings @ self.text_features.T.float()
            sim = (sim * 100).softmax(dim=-1)
            prefix_embeddings = sim @ self.text_features.float()
            prefix_embeddings /= prefix_embeddings.norm(dim=-1, keepdim=True)
        else:
            prefix_embeddings = learned_embeddings
        
        return self._decode_batch(prefix_embeddings, entry_length, temperature)
    
    def decode_from_apt_embedding(self, learned_embedding, entry_length: int = 30, temperature: float = 1.0):
        if learned_embedding.dim() == 1:
            learned_embedding = learned_embedding.unsqueeze(0)
        
        captions = self.decode_from_apt_embedding_batch(learned_embedding, entry_length, temperature)
        return captions[0] if captions else ""
    
    def decode_from_image(self, image_path: str, entry_length: int = 30, temperature: float = 1.0):
        import PIL.Image as Image
        _, preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(self.device) # type: ignore

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            if self.text_features is not None:
                sim = image_features @ self.text_features.T.float()
                sim = (sim * 100).softmax(dim=-1)
                prefix_embedding = sim @ self.text_features.float()
                prefix_embedding /= prefix_embedding.norm(dim=-1, keepdim=True)
            else:
                prefix_embedding = image_features
            
            generated_text = self._decode_single(prefix_embedding, entry_length, temperature)
            generated_text = generated_text.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
            
        return generated_text


def decode_apt_embedding(learned_embedding, 
                        model_weights_path: str = './coco_prefix-009.pt',
                        coco_train_path: str = './coco_train.json',
                        device: str = 'cuda',
                        entry_length: int = 30,
                        temperature: float = 1.0):
    decoder = APTDecoder(model_weights_path, coco_train_path, device)
    return decoder.decode_from_apt_embedding(learned_embedding, entry_length, temperature)