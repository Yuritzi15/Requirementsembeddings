# Instrucciones de entorno e instalación (CUDA 12.6)

Este proyecto usa `pandas` y `transformers` (BERT). Si tienes GPU con CUDA 12.6, sigue estas instrucciones para aprovecharla.

1) Verifica tu GPU y drivers

```powershell
nvidia-smi
```

2) Instala PyTorch con soporte CUDA 12.6

Opción A — pip (si hay wheel para cu126):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Opción B — conda (recomendado si usas conda/miniconda):

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.6 -c pytorch -c nvidia
```

3) Instala el resto de dependencias (excluyendo `torch` si ya lo instalaste con el paso anterior)

```powershell
pip install -r requirements.txt
# Si dejaste la línea de `torch` comentada en requirements.txt, este comando instalará las demás dependencias.
```

4) Verifica que PyTorch vea la GPU

```powershell
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available()); print('versión CUDA runtime:', torch.version.cuda); print('n GPUs:', torch.cuda.device_count())"
```

5) Sugerencias de rendimiento

- Usa `torch.cuda.amp` para mixed precision en entrenamiento/inferencia.
- Considera `accelerate` para facilitar despliegues y multi-GPU.

Notas:
- Revisa https://pytorch.org/get-started/locally/ si necesitas instrucciones específicas de plataforma/versión.
- Si no existe un wheel `cu126` oficial para tu plataforma, instala la versión de PyTorch recomendada por la web de PyTorch que sea compatible con tus controladores NVIDIA.

---
Archivo de dependencias: `requirements.txt`
# Requirementsembeddings