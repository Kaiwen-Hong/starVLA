"""Phase-1 probe: can the VLM ANSWER the preference when the EE-state is fed as a
learned SOFT TOKEN (not numeric text)?

Setup (diagnostic, isolated — no action stream, no DeepSpeed):
  - Frozen Qwen3-VL-4B. Only a small `phi` (per-frame EE-pose -> token embedding)
    is trainable.
  - Inject: add a `<|pref_state|>` special token; place k=n_frames of them where
    the state text used to go; a forward-hook on the embedding layer overwrites
    those positions with phi(state). The standard image / mrope path is untouched.
  - Readout: LM head answers the single discriminative token (e.g. high/low).
    Full-vocab CE on that token (faithful "VLM says the word"). `--readout probe`
    instead trains a linear classifier on the answer-position hidden state.
  - Input to phi = the SAME raw active-arm EE pose (9D x n_frames) the failed
    text path got -> clean "token vs text" isolation (phi can learn relatives).

Validation every --val_every steps: taskA-val (held-out, same objects) + taskB
(unseen objects): acc / balanced_acc / pred_dist (catches constant-collapse) / CE.
Controls: (i) zeroed state token -> must fall to ~0.50; (ii) --mlp_baseline trains
a direct MLP on the raw state (no VLM) = upper bound.

Run with starVLA python on ONE gpu. Outputs JSON + curve under r-preference/phase1/.
"""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/kaiwenh/starVLA")
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from examples.preference.dataset.vqa_sample import VQA_CATEGORIES, load_clip_and_state
from examples.preference.dataset.prompt import PREF_CATEGORIES
from examples.preference.dataset.pref_hdf5_dataset import _task_dirs_for_groups, _split_episodes

MODEL_ID = "/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct"
DATA_ROOT = Path("/mnt/localssd/kaiwenh/pref/data/0526")
OUT = Path("/home/kaiwenh/starVLA/r-preference/phase1")
IGNORE = -100
STATE_MARKER = "<|pref_state|>"
DEV = "cuda"


# ----------------------------- data -----------------------------------------

def build_pools(cat, n_val_per_leaf):
    pc = PREF_CATEGORIES[cat]
    root = DATA_ROOT / cat
    split = _split_episodes(root, pc.task_groups, pc.pref_keys, val_fraction=0.2, seed=42)
    train, vala = [], []
    for task_dir, tg, pk in _task_dirs_for_groups(root, pc.task_groups, pc.pref_keys):
        for ep in split[task_dir]["train"]:
            train.append((str(root / task_dir / "data" / f"episode{ep}.hdf5"), pk, tg))
        for ep in split[task_dir]["val"][:n_val_per_leaf]:
            vala.append((str(root / task_dir / "data" / f"episode{ep}.hdf5"), pk, tg))
    taskb = []
    tb = root / "taskB"
    if tb.is_dir():
        for d in sorted(tb.iterdir()):
            if d.is_dir() and (d / "data").is_dir():
                gt = d.name.rsplit("_", 1)[-1]
                if gt in pc.pref_keys:
                    eps = sorted(int(p.stem[7:]) for p in (d / "data").glob("episode*.hdf5"))
                    random.Random(123).shuffle(eps)
                    for ep in eps[:n_val_per_leaf]:
                        taskb.append((str(d / "data" / f"episode{ep}.hdf5"), gt, d.name))
    return pc, train, vala, taskb


def load_one(h5, strat, n, cams, jitter, rng):
    frames, state, _ = load_clip_and_state(h5, strategy=strat, n=n, cameras=cams, jitter=jitter, rng=rng)
    return frames, np.asarray(state, dtype=np.float32)        # state (n,9)


def compute_state_stats(train, strat, n, cams, k=200):
    rng = np.random.default_rng(0)
    sel = [train[i] for i in rng.choice(len(train), min(k, len(train)), replace=False)]
    S = np.stack([load_one(h5, strat, n, cams, 0, None)[1] for h5, _, _ in sel])  # (k,n,9)
    S = S.reshape(-1, S.shape[-1])
    return S.mean(0), S.std(0) + 1e-6


# ----------------------------- model -----------------------------------------

class StateTokenProbe(nn.Module):
    def __init__(self, cat, k_tokens, readout, mean, std, no_state=False):
        super().__init__()
        self.cat = cat
        self.no_state = no_state
        vc = VQA_CATEGORIES[cat]
        self.question = vc.question
        self.id_A = vc.answer_token_ids[vc.pref_keys[0]]
        self.id_B = vc.answer_token_ids[vc.pref_keys[1]]
        self.text_by_id = {self.id_A: vc.answer_text[vc.pref_keys[0]],
                           self.id_B: vc.answer_text[vc.pref_keys[1]]}
        self.pk_A, self.pk_B = vc.pref_keys
        self.cams = tuple(vc.cameras); self.strat = vc.clip_strategy; self.n = vc.n_frames
        self.k = k_tokens; self.readout = readout
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, attn_implementation="sdpa", dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.processor.tokenizer.padding_side = "left"
        self.processor.tokenizer.add_special_tokens({"additional_special_tokens": [STATE_MARKER]})
        self.vlm.resize_token_embeddings(len(self.processor.tokenizer))
        self.STATE_ID = self.processor.tokenizer.convert_tokens_to_ids(STATE_MARKER)
        self.pad_id = self.processor.tokenizer.pad_token_id
        d = self.vlm.config.text_config.hidden_size
        self.d = d

        self.phi = nn.Sequential(nn.Linear(9, 512), nn.GELU(), nn.Linear(512, d))  # per-frame
        if readout == "probe":
            self.clf = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 2))

        self.vlm.requires_grad_(False)
        self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.vlm.config.use_cache = False
        self._cur = None  # (B*k, d) state embeds for the in-flight batch
        self.vlm.get_input_embeddings().register_forward_hook(self._inject)

    def _inject(self, module, inp, out):
        if self._cur is None:
            return out
        ids = inp[0]
        mask = ids == self.STATE_ID
        out = out.clone()
        out[mask] = self._cur.to(out.dtype)
        return out

    def trainable_parameters(self):
        ps = list(self.phi.parameters())
        if self.readout == "probe":
            ps += list(self.clf.parameters())
        return ps

    def _std_state(self, state_np):  # (B,n,9) -> (B,n,9) tensor on device
        s = torch.as_tensor(np.asarray(state_np), dtype=torch.float32, device=self.mean.device)
        return (s - self.mean) / self.std

    def _messages(self, clips, answer_ids=None):
        marker = STATE_MARKER * self.k
        prompt_msgs, full_msgs = [], []
        for i, clip in enumerate(clips):
            content = [{"type": "image", "image": img} for img in clip] + \
                      [{"type": "text", "text": marker + self.question}]
            um = {"role": "user", "content": content}
            prompt_msgs.append([um])
            if answer_ids is not None:
                am = {"role": "assistant",
                      "content": [{"type": "text", "text": self.text_by_id[answer_ids[i]]}]}
                full_msgs.append([um, am])
        return prompt_msgs, full_msgs

    def _encode(self, msgs, gen):
        enc = self.processor.apply_chat_template(
            msgs, tokenize=True, padding=True, add_generation_prompt=gen,
            return_dict=True, return_tensors="pt")
        return {k: v.to(DEV) for k, v in enc.items()}

    def _set_state(self, state_std, B):
        if self.no_state:                          # pixels-only control: state carries no info
            state_std = torch.zeros_like(state_std)
        ph = self.phi(state_std)                   # (B,k,d)
        self._cur = ph.reshape(B * self.k, self.d)

    def train_loss(self, clips, states_np, answer_ids):
        B = len(clips)
        _, full_msgs = self._messages(clips, answer_ids)
        prompt_only = self._encode([[m[0]] for m in full_msgs], gen=True)
        full = self._encode(full_msgs, gen=False)
        plen = prompt_only["attention_mask"].sum(1)
        labels = full["input_ids"].clone()
        for b in range(B):
            P = int(plen[b]); labels[b, :P] = IGNORE; labels[b, P + 1:] = IGNORE
            labels[b, full["input_ids"][b] == self.pad_id] = IGNORE
        assert ((labels != IGNORE).sum(1) == 1).all(), (labels != IGNORE).sum(1).tolist()
        self._set_state(self._std_state(states_np), B)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.readout == "lm_head":
                out = self.vlm(**full, labels=labels)
                loss = out.loss
            else:
                out = self.vlm(**full, output_hidden_states=True)
                pos = (labels != IGNORE).float().argmax(1) - 1
                h = out.hidden_states[-1][torch.arange(B, device=DEV), pos]  # (B,d)
                logit2 = self.clf(h.float())
                gt = torch.tensor([0 if a == self.id_A else 1 for a in answer_ids], device=DEV)
                loss = F.cross_entropy(logit2, gt)
        # 2-way train acc
        with torch.no_grad():
            acc = self._pair_acc_from_train(out, labels, answer_ids, B)
        return loss, acc

    def _pair_acc_from_train(self, out, labels, answer_ids, B):
        if self.readout == "lm_head":
            pos = (labels != IGNORE).float().argmax(1) - 1
            al = out.logits[torch.arange(B, device=DEV), pos]
            predA = al[:, self.id_A] > al[:, self.id_B]
        else:
            pos = (labels != IGNORE).float().argmax(1) - 1
            h = out.hidden_states[-1][torch.arange(B, device=DEV), pos]
            predA = self.clf(h.float())[:, 0] > self.clf(h.float())[:, 1]
        hit = [(predA[i].item() and answer_ids[i] == self.id_A) or
               ((not predA[i].item()) and answer_ids[i] == self.id_B) for i in range(B)]
        return float(np.mean(hit))

    @torch.inference_mode()
    def predict_batch(self, clips, states_np, zero_state=False):
        B = len(clips)
        prompt_msgs, _ = self._messages(clips, None)
        enc = self._encode(prompt_msgs, gen=True)
        ss = self._std_state(states_np)
        if zero_state:
            ss = torch.zeros_like(ss)
        self._set_state(ss, B)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.readout == "lm_head":
                out = self.vlm(**enc)
                al = out.logits[:, -1, :]
                pair = torch.stack([al[:, self.id_A], al[:, self.id_B]], dim=1).float()
            else:
                out = self.vlm(**enc, output_hidden_states=True)
                h = out.hidden_states[-1][:, -1, :].float()
                pair = self.clf(h)
        p = F.softmax(pair, dim=1)
        predA = (p[:, 0] > p[:, 1]).cpu().tolist()
        return [(self.pk_A if a else self.pk_B, float(p[i, 0]), float(p[i, 1]))
                for i, a in enumerate(predA)]


# ----------------------------- eval ------------------------------------------

def evaluate(model, items, pref_keys, zero_state=False, bs=8):
    rows = []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        clips, states = [], []
        for h5, gt, tg in chunk:
            f, s = load_one(h5, model.strat, model.n, model.cams, 0, None)
            clips.append(f); states.append(s)
        preds = model.predict_batch(clips, np.stack(states), zero_state=zero_state)
        for (h5, gt, tg), (pk, pA, pB) in zip(chunk, preds):
            pc_correct = pA if gt == pref_keys[0] else pB
            pc_correct = 1e-6 if not (pc_correct == pc_correct) else pc_correct
            ce = -math.log(min(max(pc_correct, 1e-6), 1.0))
            rows.append((int(pk == gt), ce, gt, pk))
    if not rows:
        return {}
    per = {}
    for hit, _ce, gt, _pk in rows:
        per.setdefault(gt, []).append(hit)
    return {
        "acc": round(float(np.mean([r[0] for r in rows])), 3),
        "bal_acc": round(float(np.mean([np.mean(v) for v in per.values()])), 3),
        "ce": round(float(np.mean([r[1] for r in rows])), 3),
        "n": len(rows),
        "pred_dist": {k: sum(1 for r in rows if r[3] == k) for k in pref_keys},
    }


# ----------------------------- mlp baseline ----------------------------------

def mlp_baseline(cat, train, vala, taskb, pref_keys, strat, n, cams, mean, std, steps=800):
    def feats(items):
        X, y = [], []
        for h5, gt, tg in items:
            _, s = load_one(h5, strat, n, cams, 0, None)
            X.append(((s - mean) / std).reshape(-1)); y.append(0 if gt == pref_keys[0] else 1)
        return torch.tensor(np.stack(X), dtype=torch.float32), torch.tensor(y)
    Xtr, ytr = feats(train); Xa, ya = feats(vala); Xb, yb = feats(taskb)
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Linear(256, 2)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    Xtr, ytr = Xtr.to(DEV), ytr.to(DEV)
    for _ in range(steps):
        idx = torch.randint(0, len(Xtr), (128,), device=DEV)
        loss = F.cross_entropy(net(Xtr[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    def acc(X, y):
        with torch.no_grad():
            return round(float((net(X.to(DEV)).argmax(1).cpu() == y).float().mean()), 3)
    return {"taskA": acc(Xa, ya), "taskB": acc(Xb, yb), "train": acc(Xtr.cpu(), ytr.cpu())}


# ----------------------------- main ------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat", default="height")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--readout", default="lm_head", choices=["lm_head", "probe"])
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--n_val_per_leaf", type=int, default=10)
    ap.add_argument("--jitter", type=int, default=5)
    ap.add_argument("--mlp_baseline", action="store_true")
    ap.add_argument("--no_state", action="store_true", help="pixels-only control: zero the state token")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); np.random.seed(0); random.seed(0)

    pc, train, vala, taskb = build_pools(args.cat, args.n_val_per_leaf)
    vc = VQA_CATEGORIES[args.cat]
    print(f"[data] cat={args.cat} train={len(train)} taskA_val={len(vala)} taskB={len(taskb)} "
          f"pref_keys={pc.pref_keys}", flush=True)

    mean, std = compute_state_stats(train, vc.clip_strategy, vc.n_frames, tuple(vc.cameras))
    print(f"[state] mean={np.round(mean,3)} std={np.round(std,3)}", flush=True)

    if args.mlp_baseline:
        mb = mlp_baseline(args.cat, train, vala, taskb, pc.pref_keys,
                          vc.clip_strategy, vc.n_frames, tuple(vc.cameras), mean, std)
        print("[MLP upper-bound] (direct MLP on raw state, no VLM):", mb, flush=True)

    model = StateTokenProbe(args.cat, args.k, args.readout, mean, std, no_state=args.no_state).to(DEV)
    print(f"[model] readout={args.readout} k={args.k} d={model.d} STATE_ID={model.STATE_ID} "
          f"no_state={args.no_state} "
          f"trainable={sum(p.numel() for p in model.trainable_parameters())/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))

    by_pk = {pc.pref_keys[0]: [t for t in train if t[1] == pc.pref_keys[0]],
             pc.pref_keys[1]: [t for t in train if t[1] == pc.pref_keys[1]]}
    rng = random.Random(0); nrng = np.random.default_rng(0)

    def sample_batch(B):
        half = B // 2
        b = (rng.sample(by_pk[pc.pref_keys[0]], half) +
             rng.sample(by_pk[pc.pref_keys[1]], B - half))
        rng.shuffle(b)
        clips, states, ans = [], [], []
        for h5, pk, tg in b:
            f, s = load_one(h5, model.strat, model.n, model.cams, args.jitter, nrng)
            clips.append(f); states.append(s); ans.append(vc.answer_token_ids[pk])
        return clips, np.stack(states), ans

    history = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        clips, states, ans = sample_batch(args.bs)
        loss, tacc = model.train_loss(clips, states, ans)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f} | train2acc {tacc:.3f} | "
                  f"{(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            model.eval()
            a = evaluate(model, vala, pc.pref_keys)
            b = evaluate(model, taskb, pc.pref_keys)
            az = evaluate(model, vala, pc.pref_keys, zero_state=True)
            model.train()
            rec = {"step": step, "taskA": a, "taskB": b, "taskA_zeroed": az}
            history.append(rec)
            print(f"  [VAL step {step}] taskA={a} taskB={b} zeroedA={az}", flush=True)

    res = {"args": vars(args), "pref_keys": list(pc.pref_keys),
           "question": vc.question, "history": history}
    tag = args.tag or f"{args.cat}_{args.readout}"
    (OUT / f"probe_{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"\n[done] wrote {OUT / f'probe_{tag}.json'}", flush=True)
    if history:
        last = history[-1]
        print(f"[FINAL] taskA acc={last['taskA'].get('acc')} bal={last['taskA'].get('bal_acc')} | "
              f"taskB acc={last['taskB'].get('acc')} bal={last['taskB'].get('bal_acc')} | "
              f"zeroedA acc={last['taskA_zeroed'].get('acc')}", flush=True)


if __name__ == "__main__":
    main()
