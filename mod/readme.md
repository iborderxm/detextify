### 注意事项

1. **模型目录结构**：确保 `../mod` 目录中包含完整的 Stable Diffusion 2 英寸模型文件，包括：
   - `model_index.json`
   - `vae/` 目录
   - `unet/` 目录  
   - `text_encoder/` 目录
   - `scheduler/` 目录
   - 安全检查器文件等

2. **首次运行**：首次使用本地模型时，`diffusers` 库会验证模型文件完整性。如果路径无效或文件不完整，将会抛出错误。