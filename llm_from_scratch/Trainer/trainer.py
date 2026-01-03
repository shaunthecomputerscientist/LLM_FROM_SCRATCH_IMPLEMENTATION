import torch
from torch.cuda.amp import GradScaler, autocast


def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    # Ensure we are in evaluation mode
    model.eval() 
    
    for _ in range(max_new_tokens):
        # Crop context to the model's max supported length
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            # DataParallel automatically scatters 'idx_cond' across GPUs
            logits = model(idx_cond) 
            
        # Focus on the last token's output
        logits = logits[:, -1, :]

        # 1. Apply Top-K filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, 
                torch.tensor(float('-inf')).to(logits.device), 
                logits
            )

        # 2. Apply Temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# UPDATED: Removed len() calls for streaming compatibility
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    count = 0
    
    # Iterate through the loader manually
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if num_batches is not None and i >= num_batches:
            break
        
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
        count += 1
    
    # If no batches were processed, return NaN
    if count == 0:
        return float("nan")
        
    return total_loss / count

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # Disable Dropout
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train() # Re-enable Dropout
    return train_loss, val_loss

# UPDATED: Added allowed_special to prevent Tiktoken errors
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    
    # 🔍 SYSTEM 2 FIX: Access the underlying model if wrapped in DataParallel
    model_to_query = model.module if hasattr(model, "module") else model
    
    # Now this attribute access will work correctly
    context_size = model_to_query.pos_emb.weight.shape[0]
    
    # Added allowed_special to handle <|endoftext|> in high-quality corpora
    encoded = tokenizer.encode(start_context, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = generate_text_simple(
            model, encoded_tensor, max_new_tokens=15, context_size=context_size
        )
    
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(f"Sample Generation: {decoded_text.replace('\n', ' ')}")
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer, 
                       memory_efficient=True, accumulation_steps=16):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    
    # 1. Setup Scaler (Only active if memory_efficient is True)
    scaler = torch.amp.GradScaler('cuda', enabled=memory_efficient)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad() # Move outside to support accumulation
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # 2. Forward pass with Autocast
            with torch.amp.autocast('cuda', enabled=memory_efficient, dtype=torch.float16):
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                # Normalize loss for accumulation
                loss = loss / accumulation_steps 

            # 3. Scaled Backward Pass
            scaler.scale(loss).backward()
            
            tokens_seen += input_batch.numel()
            global_step += 1

            # 4. Step only after accumulation is complete
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # --- LOGGING & EVALUATION ---
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Loss {train_loss:.3f} | Val {val_loss:.3f}")
                
                generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen