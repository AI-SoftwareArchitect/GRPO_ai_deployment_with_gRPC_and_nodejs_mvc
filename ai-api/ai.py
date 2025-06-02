import json
import math
import random
import pickle
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
# REMOVE OR COMMENT OUT: from ai import * <-- This line is problematic if it's in ai.py itself.
import grpc
from concurrent import futures
import time
# Duplicated 'import torch' and 'import torch.nn.functional as F' are harmless but redundant.
# Keeping them as you included them.
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

# FIX THIS LINE:
# from chat_pb2_grpc import chat_pb2, chat_pb2_grpc  <-- This was the error cause
import chat_pb2       # Correct way to import chat_pb2
import chat_pb2_grpc  # Correct way to import chat_pb2_grpc

# CUDA ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Tokenizer:
    """Simple tokenizer for Turkish text"""
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.eos_token_id = 3  # Eğer <EOS> token'ı 3. id ise
        
    def build_vocab(self, texts: List[str], max_vocab_size: int = 8000):
        """Build vocabulary from texts"""
        word_counts = defaultdict(int)
        
        # Add special tokens
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
        
        # Count words
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] += 1
        
        # Sort by frequency and add to vocab
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words[:max_vocab_size - len(special_tokens)]:
            idx = len(self.vocab)
            self.vocab[word] = idx
            self.reverse_vocab[idx] = word
        
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size: {self.vocab_size}")
    
    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """Encode text to token ids"""
        words = text.lower().split()
        tokens = [self.vocab[self.bos_token]]
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab[self.unk_token])
        
        tokens.append(self.vocab[self.eos_token])
        
        # Pad or truncate
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.vocab[self.pad_token]] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        words = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                word = self.reverse_vocab[token_id]
                if word not in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
                    words.append(word)
        return " ".join(words)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        
        # Zero bias
        if self.W_q.bias is not None:
            nn.init.zeros_(self.W_q.bias)
        if self.W_k.bias is not None:
            nn.init.zeros_(self.W_k.bias)
        if self.W_v.bias is not None:
            nn.init.zeros_(self.W_v.bias)
        if self.W_o.bias is not None:
            nn.init.zeros_(self.W_o.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(attention_output)
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    """Single transformer block"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Multi-head attention with residual connection
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class GPTModel(nn.Module):
    """GPT-style transformer model"""
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 2048, max_length: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_length, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff) 
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize LM head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.size()
        
        # Position ids
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + pos_emb
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        probs = F.softmax(logits, dim=-1)
        
        return logits, probs
    
    def generate(self, input_ids: List[int], max_new_tokens: int = 50, 
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.2) -> List[int]:
        """Generate text using the model"""
        self.eval()
        generated = input_ids.copy()
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self, "tokenizer") else 3
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Prepare input
                current_input = torch.tensor([generated[-self.max_length:]], 
                                           dtype=torch.long, device=device)
                
                # Forward pass
                logits, probs = self.forward(current_input)
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :] / temperature

                # Repetition penalty
                for token_id in set(generated):
                    next_token_logits[token_id] /= repetition_penalty

                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token_probs = F.softmax(filtered_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, num_samples=1).item()
                generated.append(next_token)
                
                # Stop if EOS token
                if next_token == eos_token_id:
                    break
        
        return generated

def top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-float('Inf')):
    # Top-k
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    # Top-p (nucleus)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

class TextDataset(Dataset):
    """Custom dataset for text generation"""
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = item['prompt'] + " " + item['response']
        tokens = self.tokenizer.encode(full_text, self.max_length)
        return torch.tensor(tokens, dtype=torch.long)

class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) with Self-Certainty"""
    def __init__(self, model: GPTModel, tokenizer: Tokenizer, 
                 learning_rate: float = 1e-4, gamma: float = 0.99, 
                 epsilon: float = 0.2, batch_size: int = 4, accumulation_steps: int = 4):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_history = []
        self.reward_history = []
        self.perplexity_history = []
        self._step = 0

    def compute_self_certainty(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Compute self-certainty as reward signal (batch)"""
        # logits: [batch, seq, vocab], target_ids: [batch, seq]
        probs = F.softmax(logits, dim=-1)
        target_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        mask = (target_ids != 0).float()
        log_probs = torch.log(target_probs + 1e-8) * mask
        # Average over valid tokens for each batch
        certainty = log_probs.sum(dim=1) / mask.sum(dim=1)
        return certainty  # shape: [batch]

    def compute_advantage(self, rewards: List[float]) -> List[float]:
        """Compute advantages using GAE (Generalized Advantage Estimation)"""
        advantages = []
        baseline = np.mean(rewards)
        for reward in rewards:
            advantages.append(reward - baseline)
        return advantages
    
    def policy_loss(self, old_probs: torch.Tensor, new_probs: torch.Tensor, 
                   advantages: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """Compute GRPO policy loss"""
        old_target_probs = old_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        new_target_probs = new_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        mask = (target_ids != 0).float()
        ratio = (new_target_probs / (old_target_probs + 1e-8)) * mask
        # Broadcast advantages to [batch, seq]
        advantages = advantages.unsqueeze(1)  # [batch, 1]
        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        policy_loss = -torch.min(loss1, loss2) * mask
        return policy_loss.sum() / mask.sum()
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        batch = batch.to(device)
        logits, probs = self.model(batch)
        rewards = self.compute_self_certainty(logits[:, :-1], batch[:, 1:])  # shape: [batch]
        old_probs = probs.detach()
        advantages = rewards  # shape: [batch]
        loss = self.policy_loss(old_probs[:, :-1], probs[:, :-1], advantages, batch[:, 1:])
        loss = loss / self.accumulation_steps  # Gradient accumulation için loss'u böl

        loss.backward()
        self._step += 1

        if self._step % self.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        with torch.no_grad():
            mask = (batch != 0).float()
            log_probs = torch.log(probs.gather(2, batch.unsqueeze(-1)).squeeze(-1) + 1e-8) * mask
            perplexity = torch.exp(-log_probs.sum() / mask.sum()).item()
        self.loss_history.append(loss.item() * self.accumulation_steps)
        self.reward_history.append(rewards.detach().cpu().tolist())
        self.perplexity_history.append(perplexity)
        return {
            'loss': loss.item() * self.accumulation_steps,
            'reward': float(rewards.mean().item()),
            'perplexity': perplexity
        }

    def train(self, data: List[Dict], num_epochs: int = 10):
        """Train the model using GRPO"""
        print(f"Starting GRPO training for {num_epochs} epochs...")
        dataset = TextDataset(data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_metrics = {'loss': [], 'reward': [], 'perplexity': []}
            for batch in dataloader:
                metrics = self.train_step(batch)
                for key in epoch_metrics:
                    epoch_metrics[key].append(metrics[key])
            # Düzeltme: rewardlar artık float, doğrudan np.mean kullanılabilir
            avg_loss = np.mean(epoch_metrics['loss'])
            avg_reward = np.mean(epoch_metrics['reward'])
            avg_perplexity = np.mean(epoch_metrics['perplexity'])
            print(f"Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}, Perplexity: {avg_perplexity:.4f}")

    def plot_training_history(self):
        """Plot training metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        # Düzeltme: reward_history içindeki listeleri düzleştir
        flat_rewards = [item for sublist in self.reward_history for item in (sublist if isinstance(sublist, list) else [sublist])]
        ax2.plot(flat_rewards)
        ax2.set_title('Self-Certainty Reward')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward')
        ax2.grid(True)
        ax3.plot(self.perplexity_history)
        ax3.set_title('Perplexity')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Perplexity')
        ax3.grid(True)
        if self.loss_history and flat_rewards and self.perplexity_history:
            normalized_loss = np.array(self.loss_history) / np.max(self.loss_history)
            normalized_reward = np.array(flat_rewards) / np.max(np.abs(flat_rewards))
            normalized_perplexity = np.array(self.perplexity_history) / np.max(self.perplexity_history)
            ax4.plot(normalized_loss, label='Loss (norm)', alpha=0.7)
            ax4.plot(normalized_reward, label='Reward (norm)', alpha=0.7)
            ax4.plot(normalized_perplexity, label='Perplexity (norm)', alpha=0.7)
            ax4.set_title('Normalized Metrics')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Normalized Value')
            ax4.legend()
            ax4.grid(True)
        plt.tight_layout()
        plt.show()

# Random seed'ler
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_data(file_path: str) -> List[Dict]:
    """Load training data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} training examples")
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found!")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}")
        return []

def save_model(model: GPTModel, tokenizer: Tokenizer, filepath: str):
    """Save model and tokenizer"""
    model_data = {
        'model_state': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocab,
        'tokenizer_reverse_vocab': tokenizer.reverse_vocab,
        'tokenizer_vocab_size': tokenizer.vocab_size,
        'config': {
            'vocab_size': model.vocab_size,
            'd_model': model.d_model,
            'num_heads': model.num_heads,
            'num_layers': model.num_layers,
            'max_length': model.max_length,
            'd_ff': model.blocks[0].feed_forward.linear1.out_features  # d_ff'i ekle
        }
    }
    torch.save(model_data, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: str) -> Tuple[GPTModel, Tokenizer]:
    """Load model and tokenizer"""
    try:
        model_data = torch.load(filepath, map_location=device)
        tokenizer = Tokenizer()
        tokenizer.vocab = model_data['tokenizer_vocab']
        tokenizer.reverse_vocab = model_data['tokenizer_reverse_vocab']
        tokenizer.vocab_size = model_data['tokenizer_vocab_size']
        config = model_data['config']
        model = GPTModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config.get('d_ff', 1024),  # d_ff parametresini ekle
            max_length=config['max_length']
        )
        model.load_state_dict(model_data['model_state'])
        model.to(device)
        print(f"Model loaded from {filepath}")
        return model, tokenizer
    except FileNotFoundError:
        print(f"Model file {filepath} not found!")
        return None, None

def train_model():
    print("=" * 50)
    print("GRPO TRANSFORMER TRAINING")
    print("=" * 50)
    data = load_data('data.json')
    if not data:
        print("No training data found!")
        return
    all_texts = []
    for item in data:
        all_texts.append(item['prompt'])
        all_texts.append(item['response'])
    print("Building vocabulary...")
    tokenizer = Tokenizer()
    tokenizer.build_vocab(all_texts, max_vocab_size=8000)
    print("Initializing model...")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,           # Küçük model, VRAM dostu
        num_heads=4,
        num_layers=4,
        d_ff=1024,
        max_length=128
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.1f}M")
    trainer = GRPOTrainer(
        model, tokenizer,
        learning_rate=3e-5,   # Düşük learning rate, stabil eğitim
        batch_size=4,         # Küçük batch, VRAM dostu
        accumulation_steps=16 # Efektif batch büyüt
    )
    num_epochs = 20          # Yeterli epoch, overfitten kaçınmak için erken durdurma ekleyebilirsin
    trainer.train(data, num_epochs=num_epochs)
    save_model(model, tokenizer, 'grpo_model.pt')
    print("\nTraining completed! Showing performance plots...")
    trainer.plot_training_history()
    print("Model saved as 'grpo_model.pt'")

def test_model():
    """Testing/interaction procedure"""
    print("=" * 50)
    print("GRPO TRANSFORMER TESTING")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model('grpo_model.pt')
    if model is None:
        print("No trained model found! Please train first.")
        return
    
    print("Model loaded successfully!")
    print("Type 'quit' to exit testing mode.")
    print("-" * 30)
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        try:
            # Encode input
            input_tokens = tokenizer.encode(user_input, max_length=64)
            
            # Generate response
            generated_tokens = model.generate(
                input_tokens, 
                max_new_tokens=50, 
                temperature=1.0, 
                top_k=50, 
                top_p=0.95,
                repetition_penalty=1.2
            )
            
            # Decode response
            response = tokenizer.decode(generated_tokens[len(input_tokens):])
            
            print(f"AI: {response}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error generating response: {e}")
            print("-" * 30)