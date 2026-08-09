# 3D Object Project

Tek foto ya da birkaç kanonik açıdan foto girdisinden **3D obje** üreten bir sistem.
Objaverse veri seti → Blender ile çok-görünümlü render → transformer-tabanlı LRM eğitimi → web app.

## Durum

- [x] Veri indirildi (51.534 `.glb`, Objaverse)
- [x] Mimari kararları (bkz. `CLAUDE.md`)
- [ ] **Faz A — Render pipeline** (aktif)
- [ ] Faz B — LRM eğitimi
- [ ] Faz C — Web app

## Hedef

Kullanıcı 1 foto (ör. "ön") ya da 4 kanonik açı (ön/arka/sol/sağ) yükler →
model 3D mesh (`.glb`) üretir → tarayıcıda three.js ile görüntülenir.

## Yapı

```
3d_object_project/
├── CLAUDE.md              # Proje bağlamı + tüm kararlar (önce bunu oku)
├── README.md
├── docs/
│   └── superpowers/specs/ # Tasarım dokümanları (spec'ler)
├── scripts/               # Faz A: render scriptleri (Blender)
├── src/                   # Faz B/C: model + web app kodu
└── dataset/               # Render çıktıları (git'e girmez)
```

## Ortam

- GPU: RTX 5080 (16 GB, Blackwell) → CUDA 12.8+ / güncel PyTorch gerekir
- Windows 11, Python 3.10, Blender 4.4

Detaylar ve kararların gerekçesi için **`CLAUDE.md`**'ye bak.
