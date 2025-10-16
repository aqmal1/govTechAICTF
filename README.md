# AI CTF 2025 Write-up

## KopiTalk (Aqmal)

![KopiTalk Screenshot](img/media/image1.png)

Being a coffee addict, I recorded myself saying "kopi peng", which is my default order, and uploaded it. However, I could not get the **Healthy tokens** option to be true. So, I uploaded myself saying "auntie kopi o kosong auntie kopi o kosong" and got the flag!

![Flag Result](img/media/image2.png)

---

## MNIST (Lucian)

### Code Given

```python
import os, io, base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# parameters
k, l, n, q = 8, 8, 3, 256

def make_bit_indices(k, l):
    need = k + l
    coords = []
    for r in range(28):
        for c in range(28):
            coords.append(r*28 + c)
    return np.array(coords[:need], dtype=np.int64)

bit_idx = make_bit_indices(k, l)
print("bit_idx length:", len(bit_idx), "first indices:", bit_idx[:8])

buffers_path = "buffers.npz"  # put the original buffers.npz here
if not os.path.exists(buffers_path):
    raise FileNotFoundError(f"Place buffers.npz (from original notebook) in this folder as {buffers_path}")

buf = np.load(buffers_path)
B = buf['B']  # expected shape (n, k*l), e.g. (3,64)
y = buf['y']  # expected shape (n,)
print("Loaded buffers:", "B.shape=", B.shape, "y.shape=", y.shape)

m_star = np.ones(k, dtype=np.int64)

def bits_from_int(x, length):
    # little-endian bit order (matches original helper)
    return np.array([(x >> i) & 1 for i in range(length)], dtype=np.int64)

s_found = None
for val in range(256):
    s_bits = bits_from_int(val, l)
    mt = np.outer(m_star, s_bits).reshape(-1)  # flatten by rows
    lhs = (B.dot(mt) % q).astype(np.int64)
    rhs = (y % q).astype(np.int64)
    if np.array_equal(lhs, rhs):
        s_found = s_bits
        print("Found s (bits little-endian):", s_found, "int value:", val)
        break

if s_found is None:
    raise RuntimeError("Could not find s. Check buffers.npz.")

def stamp_bits_into_image_np(img_np, bit_idx, m_bits, s_bits):
    """
    img_np: (28,28) uint8 (0..255)
    bit_idx: array length k+l with positions 0..783
    m_bits: length k
    s_bits: length l
    Returns uint8 (28,28)
    """
    out = img_np.reshape(-1).copy()
    bits = np.concatenate([m_bits, s_bits]).astype(np.int64)
    # write bright (255) for 1, dark (0) for 0.
    # After ToTensor+Normalize used by the model, these map to >0.5 and <=0.5 respectively.
    for i, idx in enumerate(bit_idx):
        out[idx] = 255 if bits[i] == 1 else 0
    return out.reshape(28, 28).astype(np.uint8)

# prepare m and s
m_bits = m_star.astype(np.int64)  # all ones
s_bits = s_found.astype(np.int64)

# blank canvas (black background)
blank = np.zeros((28, 28), dtype=np.uint8)
blank_stamped = stamp_bits_into_image_np(blank, bit_idx, m_bits, s_bits)
Image.fromarray(blank_stamped).save("backdoor_blank.png")
print("Saved backdoor_blank.png (blank image with stamp)")

# show inline
plt.figure(figsize=(3, 3))
plt.imshow(blank_stamped, cmap='gray')
plt.title("backdoor_blank.png")
plt.axis('off')
plt.show()

def image_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

b64_blank = image_to_b64("backdoor_blank.png")
with open("backdoor_blank.b64.txt", "w") as f:
    f.write(b64_blank)

print("Wrote base64 to backdoor_blank.b64.txt (length {})".format(len(b64_blank)))
print("First 200 chars of base64:\n", b64_blank[:200])
```

### Submission Code

```python
import json, hashlib, time, base64, os
from urllib.parse import urljoin

try:
    import requests
except Exception as e:
    raise RuntimeError("requests is required. Run Cell 1 to install it.") from e

def get_challenge(server):
    url = urljoin(server.rstrip('/') + '/', 'challenge')
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def solve_pow(chal_str, difficulty, max_tries=None):
    """Find nonce (string of integer) such that sha256(nonce + chal_str) startswith '0'*difficulty.
    This is simple incremental search; prints progress every 100k tries.
    """
    need = "0" * int(difficulty)
    n = 0
    start = time.time()
    while True:
        nonce = str(n)
        h = hashlib.sha256((nonce + chal_str).encode()).hexdigest()
        if h.startswith(need):
            dt = time.time() - start
            print(f"Found nonce={nonce} after {n+1:,} tries in {dt:.2f}s")
            return nonce
        n += 1
        if max_tries and n >= max_tries:
            raise RuntimeError("Reached max_tries without finding nonce.")
        if (n % 100000) == 0:
            elapsed = time.time() - start
            rate = n / elapsed if elapsed > 0 else 0
            print(f"tried {n:,} nonces --- {rate:,.0f}/s")

def post_infer(server, challenge_id, nonce, image_b64):
    url = urljoin(server.rstrip('/') + '/', 'infer')
    payload = {"challenge_id": challenge_id, "nonce": nonce, "image": image_b64}
    r = requests.post(url, json=payload, timeout=60)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text

def load_image_base64(path=None, b64file=None):
    """Load base64 string either from an existing .b64.txt or by encoding the PNG file."""
    if b64file and os.path.exists(b64file):
        with open(b64file, "r") as f:
            return f.read().strip()
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    raise FileNotFoundError("Provide either existing .b64.txt file or the PNG path.")

def submit_backdoor(server, image_path="backdoor_blank.png", b64file=None, max_pow_tries=None):
    print("Requesting challenge from server:", server)
    chal = get_challenge(server)
    cid = chal.get("challenge_id") or chal.get("challengeId") or chal.get("id")
    challenge_str = chal.get("challenge") or chal.get("chal") or chal.get("challenge_str")
    difficulty = chal.get("difficulty", 0)
    
    if cid is None or challenge_str is None:
        raise RuntimeError(f"Unexpected /challenge response: {chal}")
    
    print("Challenge id:", cid, "difficulty:", difficulty)
    print("Solving PoW --- this may take time depending on difficulty...")
    nonce = solve_pow(challenge_str, difficulty, max_tries=max_pow_tries)
    
    print("Loading image base64...")
    img_b64 = load_image_base64(path=image_path, b64file=b64file)
    
    print("Posting inference...")
    status, body = post_infer(server, cid, nonce, img_b64)
    print("Response status:", status)
    print("Response body:")
    print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
    return status, body

# Main execution
SERVER = "https://mnist.aictf.sg/"
IMAGE_PATH = "backdoor_blank.png"
B64_FILE = "backdoor_blank.b64.txt"

print("Files in current directory:", [f for f in os.listdir('.') if f.startswith('backdoor')])

if not os.path.exists(IMAGE_PATH) and not os.path.exists(B64_FILE):
    raise FileNotFoundError("Cannot find image or base64 file.")

try:
    status, body = submit_backdoor(SERVER, image_path=IMAGE_PATH, b64file=B64_FILE)
    print("\nFinal result from server:")
    print("HTTP status:", status)
    if isinstance(body, dict):
        print(json.dumps(body, indent=2))
    else:
        print(body)
except Exception as e:
    print("Submission failed:", e)
```

![MNIST Result](img/media/image3.png)

---

## StrideSafe (Delton)

![StrideSafe Challenge](img/media/image4.png)

Used OpenAI CLIP to classify human images vs scooter images. Plot the classification which will show a QR code.

**Flag:** `AI2025{5tr1d3s4f3_15_l1t}`

![QR Code 1](img/media/image5.png)
![QR Code 2](img/media/image6.png)

![StrideSafe Result](img/media/image7.png)

---

## CoPirate (Loysius)

![CoPirate Challenge](img/media/image8.png)

---

## Guardrails (Loysius)

![Guardrails Challenge](img/media/image9.png)

![Guardrails Result](img/media/image10.png)

---

## Deepfake Detection (Loysius)

I (Aqmal) found out that it just submitted your answers in an array of 0s and 1s to the endpoint and you need to get a high enough score to get the flag. So I did that but for the life of me could not get the answer. So I did it manually with 2 tabs open, one is the CTF and the other is a deepfake analyser website. AND YET I STILL COULDN'T GET THE FLAG! However, my teammate, Loysius found another website which is better and did it manually (uploading each image one by one) and managed to get the flag. I guess some things really can only be solved by force.

![Deepfake Detection Result](img/media/image11.png)

![Moon Knight Meme](img/media/image12.jpeg)

![Deepfake Process](img/media/image13.png)

![Deepfake Flag](img/media/image14.png)

---

## Lion Roar (Delton)

To unlock the secret flag, I need to have a special key.

So the prompt should be related to the special key instead of the secret flag.

Prompt the LLM to get information regarding Tony Chua from the LinkedIn profile of Vipul G. until we find the link of LionMind-GPT. In the activity, we can find the hidden API Key from his past commit.

![Lion Roar Challenge](img/media/image15.png)

![Lion Roar Commit](img/media/image16.png)

![Lion Roar Result](img/media/image17.png)

---

## Doctor Chao Keng (Loysius)

Find the extra parameter which is "ad" and try to prompt until Doctor gives mc.

![Doctor Chao Keng Challenge](img/media/image18.png)

### Code Snippet

```typescript
import { useState, useRef, useEffect } from "react";
import { toast } from "@/hooks/use-toast";
import ChatMessage from "@/components/ChatMessage";
import SymptomForm, { FormData } from "@/components/SymptomForm";
import pokemonCenterBg from "@/assets/pokemon-center-bg.png";

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const Index = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      text: "Welcome to Doctor CK's virtual clinic! I'm here to help you get your MC today. Please share your symptoms below!",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionClosed, setSessionClosed] = useState(false);
  const [isAD, ad] = useState(false);
  const [musicStarted, setMusicStarted] = useState(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const videoId = "M2B7wSxwDcQ";
  const videoSrc = `https://www.youtube.com/embed/${videoId}?autoplay=1&loop=1&playlist=${videoId}&controls=0&showinfo=0&rel=0&iv_load_policy=3&modestbranding=1&mute=${musicStarted ? 0 : 1}`;

  const scrollToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  const handleInteraction = () => {
    if (!musicStarted) {
      setMusicStarted(true);
    }
  };

  useEffect(() => {
    window.addEventListener('click', handleInteraction);
    return () => {
      window.removeEventListener('click', handleInteraction);
    };
  }, [musicStarted]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFormSubmit = async (formData: FormData) => {
    const userMessage: Message = raiseTicket(formData);
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const payload = Object.fromEntries(
        Object.entries(formData)
          .filter(([key, value]) => key !== 'ad' || (key === 'ad' && value))
      );

      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Failed to get response from server");
      }

      const data = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.response,
        isUser: false,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);

      if (data.mc && typeof data.mc === 'string') {
        const mcMessage: Message = {
          id: (Date.now() + 2).toString(),
          text: "Medical Certificate: " + data.mc,
          isUser: false,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, mcMessage]);
      }

      toast({
        title: "Consultation Ended.",
        description: "Did you get your MC? 😉",
        variant: "default"
      });

      setSessionClosed(true);
    } catch (error) {
      console.error("Error:", error);
      toast({
        title: "Connection Error",
        description: "Unable to connect to the consultation server. Please try again in a while!",
        variant: "destructive",
      });

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: "I apologise, but I'm having trouble connecting to the server. Please visit DoctorCK again later!",
        isUser: false,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // todo: allow common symptoms only first ba.
  // still trying to add my guardrails proper (though it can already defend trivial/medium ones.. i think)
  // keep func as shortcut for now lol.
  useEffect(() => {
    (window as any).peeka = () => {
      ad(prev => !prev);
    };
    return () => { delete (window as any).peeka; };
  }, []);

  const raiseTicket = (formData: FormData): Message => {
    return {
      id: Date.now().toString(),
      text: `Name: ${formData.name}\nSymptoms: ${formData.symptoms.length > 0 ? formData.symptoms.join(", ") : "None selected"}\nAdditional Details: ${formData.ad || "-"}`,
      isUser: true,
      timestamp: new Date(),
    };
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary via-background to-secondary/30 relative">
      {/* Background Music Player */}
      {musicStarted && (
        <iframe
          src={videoSrc}
          style={{ display: 'none' }}
          allow="autoplay"
        />
      )}

      {/* 8-bit Background */}
      <div
        className="absolute inset-0 opacity-20 bg-center"
        style={{
          backgroundImage: `url(${pokemonCenterBg})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          imageRendering: 'pixelated',
          filter: 'contrast(1.2)'
        }}
      />

      <div className="container mx-auto px-4 py-8 max-w-4xl relative z-10">
        {/* Header */}
        <header className="text-center mb-8">
          <div
            className="inline-block bg-primary text-primary-foreground px-8 py-6 border-4 border-primary/80 mb-4"
            style={{
              boxShadow: '8px 8px 0px rgba(0, 0, 0, 0.3)',
              borderRadius: '0px'
            }}
          >
            <div className="flex items-center justify-center gap-3 mb-2">
              <span className="text-2xl animate-blink">💉</span>
              <h1 className="text-xl font-bold uppercase tracking-wider">
                Doctor CK
              </h1>
              <span className="text-2xl animate-blink" style={{ animationDelay: '0.7s' }}>💉</span>
            </div>
            <p className="text-[10px] uppercase tracking-wide mt-2">
              Get Your MCs Today (only if you are really sick!)
            </p>
          </div>
        </header>

        {/* Chat Container */}
        <main
          className="bg-card border-4 border-border overflow-hidden"
          style={{
            boxShadow: '8px 8px 0px rgba(0, 0, 0, 0.3)',
            borderRadius: '0px'
          }}
        >
          <div
            ref={chatContainerRef}
            className="h-[400px] overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-secondary/10 to-background"
            style={{
              backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,.03) 2px, rgba(0,0,0,.03) 4px)'
            }}
          >
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message.text}
                isUser={message.isUser}
                timestamp={message.timestamp}
              />
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div
                  className="bg-secondary text-secondary-foreground border-4 border-secondary/80 px-4 py-3 animate-pulse-glow"
                  style={{
                    boxShadow: '4px 4px 0px rgba(0, 0, 0, 0.3)',
                    borderRadius: '0px'
                  }}
                >
                  <div className="flex space-x-2 items-center">
                    <div className="w-2 h-2 bg-primary animate-bounce" style={{ borderRadius: '0px' }} />
                    <div className="w-2 h-2 bg-primary animate-bounce" style={{ animationDelay: '0.1s', borderRadius: '0px' }} />
                    <div className="w-2 h-2 bg-primary animate-bounce" style={{ animationDelay: '0.2s', borderRadius: '0px' }} />
                    <span className="text-[8px] ml-2 animate-blink">PROCESSING...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Form Section */}
          <div className="p-6 bg-muted/50 border-t-4 border-border">
            <SymptomForm
              onSubmit={handleFormSubmit}
              disabled={isLoading || sessionClosed}
              showOthersField={isAD}
            />
          </div>
        </main>

        {/* Footer */}
        <footer className="text-center mt-6 space-y-2">
          <br /> <br />
          <p
            className="inline-block bg-accent text-accent-foreground px-4 py-2 border-2 border-accent/80 text-[8px] uppercase font-bold"
            style={{
              boxShadow: '4px 4px 0px rgba(0, 0, 0, 0.2)',
              borderRadius: '0px'
            }}
          >
            Emergency? Call 995 or please see a doctor immediately!
          </p>
          <br /> <br /> <br />
          <p className="text-[8px] text-muted-foreground">
            BGM: <a
              href="https://youtube.com/@CinderyLofi"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-primary">Cindery</a>
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Index;
```

![Doctor Chao Keng Result](img/media/image19.png)

---

## Fool the FashionNet! (Aqmal)

Not gonna lie, I just used Deepseek and kept prompting until I got the flag HAHAHA.

![Fool the FashionNet Result](img/media/image20.png)

### Attack Script

Assuming the name of the file is `attack.py`:

```bash
python attack.py --step_size 0.02 --steps 2000 --restarts 20 --target_class 6
```

Output is a `delta.npy` file.

```python
#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SmallFashionCNN
from utils import linf, ssim_01

def load_image_28x28_gray(path):
    x = np.asarray(Image.open(path).convert("L").resize((28, 28)), dtype=np.float32) / 255.0
    return x[None, None, :, :]

def load_model(weights_path):
    m = SmallFashionCNN(num_classes=10)
    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "model_state" in sd:
        sd = sd["model_state"]
    try:
        m.load_state_dict(sd, strict=True)
    except Exception:
        m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    m.eval()
    return m

def clamp01(t):
    return torch.clamp(t, 0.0, 1.0)

MEAN = 0.2860
STD = 0.3530

def normalize(x):
    return (x - MEAN) / STD

def simple_pgd_attack(model, x, target_class, eps=0.08, steps=1000, step_size=0.01):
    """Simple but aggressive PGD that guarantees flip"""
    x_adv = x.clone().detach()
    for i in range(steps):
        x_adv.requires_grad = True
        logits = model(normalize(x_adv))
        # Use simple cross-entropy loss for more aggressive attack
        loss = F.cross_entropy(logits, torch.tensor([target_class]))
        grad = torch.autograd.grad(loss, x_adv)[0]
        # Sign gradient for L∞ attack
        x_adv = x_adv.detach() - step_size * grad.sign()
        # Project back to L∞ ball and [0,1] range
        delta = torch.clamp(x_adv - x, -eps, eps)
        x_adv = torch.clamp(x + delta, 0, 1)
        
        # Check if we've succeeded
        with torch.no_grad():
            pred = model(normalize(x_adv)).argmax(1).item()
            if pred == target_class:
                delta_np = (x_adv - x).detach().cpu().numpy()
                max_norm = np.max(np.abs(delta_np))
                if max_norm <= eps:
                    a = x_adv.detach().cpu().numpy()[0, 0]
                    b = x.detach().cpu().numpy()[0, 0]
                    ssim_val = ssim_01(b, a)
                    return True, delta_np.reshape(1, 28, 28), {
                        "orig": model(normalize(x)).argmax(1).item(),
                        "adv": pred, "ssim": ssim_val, "linf": max_norm
                    }
    return False, None, {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_img", default="seed.png")
    ap.add_argument("--weights", default="model.pt")
    ap.add_argument("--out", default="delta.npy")
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--step_size", type=float, default=0.01)
    ap.add_argument("--target_class", type=int, default=6)
    ap.add_argument("--restarts", type=int, default=10)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    x_np = load_image_28x28_gray((here / args.seed_img).as_posix())
    x = torch.from_numpy(x_np).float()
    model = load_model((here / args.weights).as_posix()).eval()

    with torch.no_grad():
        orig = model(normalize(x)).argmax(1).item()
    print(f"[+] Original prediction: {orig}")
    print(f"[+] Target: flip to class {args.target_class}")

    for r in range(args.restarts):
        ok, dnp, info = simple_pgd_attack(
            model, x, args.target_class,
            eps=args.eps - 1e-6,  # Strict inequality
            steps=args.steps,
            step_size=args.step_size
        )
        if ok and info["ssim"] >= 0.85:
            np.save((here / args.out).as_posix(), dnp.astype(np.float32))
            print(f"[✓] Success! Saved {args.out}")
            print(f"  {info['orig']} → {info['adv']}, "
                  f"SSIM={info['ssim']:.4f}, L∞={info['linf']:.6f}")
            return
        elif ok:
            print(f"  - Restart {r+1}: flip OK but SSIM={info['ssim']:.4f} (need ≥0.85)")
        else:
            print(f"  - Restart {r+1}: no flip")

    print("[x] Failed to find valid perturbation")
    print("[!] Try increasing --steps or --step_size")

if __name__ == "__main__":
    main()
```

---

## MLMPire (Aqmal)

This one also I keep using Chat GPT until I got the answer.

![MLMPire Result](img/media/image21.png)

### Hunt Flag Script

```bash
python hunt_flag.py --beam_k 16 --beam_steps 220
```

```python
# hunt_flag.py
import argparse, math, re, torch
import torch.nn.functional as F
from architecture import load_hf_gpt2_model, SEQ_LEN

FLAG_START = "AI2025{"
ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.:,/@!#$%^&*")  # extend if needed

def logits_at_mask(wrapper, s):
    ids = wrapper.encode(s + "[MASK]", seq_len=SEQ_LEN)
    attn = (ids != wrapper.stoi["[PAD]"]).long()
    out = wrapper.model(input_ids=ids, attention_mask=attn)
    pos = ids[0].tolist().index(wrapper.stoi["[MASK]"])
    return out[0][0, pos]

def topk_chars(wrapper, logits, k=10):
    vals, idxs = torch.topk(F.softmax(logits, dim=-1), k)
    # map to tokens
    itos = wrapper.itos
    toks = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        t = itos[i] if isinstance(itos, dict) else itos(i)
        toks.append((t, v, i))
    return toks

def constrained_finish(wrapper, prefix, max_inside=120, verbose=True):
    """Continue AFTER seeing AI2025{, allowing only ALLOWED or '}'."""
    itos = wrapper.itos
    inside = ""
    while len(inside) < max_inside:
        logits = logits_at_mask(wrapper, prefix + inside)
        # build mask
        mask = torch.full_like(logits, float("-inf"))
        allowed_ids = []
        for tid in range(len(wrapper.vocab)):
            tok = itos[tid] if isinstance(itos, dict) else itos(tid)
            if tok == "}" or (len(tok) == 1 and tok in ALLOWED):
                allowed_ids.append(tid)
        mask[torch.tensor(allowed_ids)] = 0.0
        logits = logits + mask
        vals, idxs = torch.topk(F.softmax(logits, dim=-1), 10)
        for j, tid in enumerate(idxs.tolist()):
            tok = itos[tid] if isinstance(itos, dict) else itos(tid)
            if tok == "}" or (len(tok) == 1 and tok in ALLOWED):
                inside += tok
                if verbose:
                    print(f"  + '{tok}' (p={vals[j]:.3f}) => {FLAG_START+inside}")
                break
        if inside.endswith("}"):
            return FLAG_START + inside
    return None

def beam_search_until_flag(wrapper, seeds, steps=160, k=12):
    beams = [(s, 0.0) for s in seeds]  # (text, logprob)
    seen = set(s for s, _ in beams)
    for t in range(steps):
        new_beams = []
        for s, lp in beams:
            logits = logits_at_mask(wrapper, s)
            for tok, p, _ in topk_chars(wrapper, logits, k=k):
                if tok in ("[PAD]", "[UNK]") or tok.startswith("["):
                    continue
                ns = s + tok
                if ns in seen:
                    continue
                seen.add(ns)
                nlp = lp + math.log(p + 1e-12)
                new_beams.append((ns, nlp))
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:k]
        # check for start
        for s, _ in beams:
            i = s.find(FLAG_START)
            if i != -1:
                return s[:i+len(FLAG_START)]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default=None, help="Full context ending with AI2025{")
    ap.add_argument("--seeds", nargs="*", default=[
        "TRY: action=post path=/api/auth/login type=request body={\\\"name\\\":\\\"admin\\\",\\\"pw\\\":\\\"",
        "ENTRY: action=post path=/api/auth/login type=request body={\\\"name\\\":\\\"admin\\\",\\\"pw\\\":\\\"",
        "token=", "password=", "pwd="
    ])
    ap.add_argument("--beam_k", type=int, default=12)
    ap.add_argument("--beam_steps", type=int, default=160)
    ap.add_argument("--max_inside", type=int, default=120)
    args = ap.parse_args()

    wrapper = load_hf_gpt2_model(model_path=".", vocab_path="./vocab.json")
    print(f"[+] Loaded; vocab={len(wrapper.vocab)} seq_len={SEQ_LEN}")

    prefix = args.prefix
    if prefix and not prefix.endswith(FLAG_START):
        if not prefix.endswith('"'):
            prefix += '"'
        if not prefix.endswith(FLAG_START):
            prefix += FLAG_START

    if not prefix:
        print("[i] Searching for the flag start with seeds...")
        prefix = beam_search_until_flag(wrapper, args.seeds, 
                                        steps=args.beam_steps, k=args.beam_k)
        if not prefix:
            print("[x] Could not reach AI2025{ from seeds; raise --beam_steps/--beam_k or adjust seeds.")
            return

    print(f"[+] Found start context:\n{prefix[-120:]}")
    print("[i] Constraining characters until '}' ...")
    flag = constrained_finish(wrapper, prefix, 
                             max_inside=args.max_inside, verbose=True)

    if flag:
        print(f"\n[✓] FLAG: {flag}")
    else:
        print("[x] Reached cap without closing brace. Try a different seed/context or raise --max_inside.")

if __name__ == "__main__":
    main()
```

---

## Summary

This CTF covered various AI security challenges including:

- **Audio/Speech Recognition** - KopiTalk
- **Computer Vision & Backdoors** - MNIST, StrideSafe, Fool the FashionNet!
- **Deepfake Detection** - Manual analysis challenge
- **LLM Security** - CoPirate, Guardrails, Lion Roar, Doctor Chao Keng, MLMPire
- **Adversarial Attacks** - FashionNet perturbation generation

Key techniques used:
- CLIP for image classification
- PGD attacks for adversarial examples
- Beam search for flag extraction from language models
- Manual deepfake analysis
- API key discovery through OSINT
- Prompt engineering and jailbreaking

All challenges required creative approaches combining ML knowledge with security techniques!
