## Vitis AI workflow (ZCU104)

```bash
# 1) Float evaluation (no quantization)
python eval_and_quantize_seg3.py --ckpt <CKPT> --quant_mode float

# 2) PTQ calibration (writes quant artifacts to quantize_result/)
python eval_and_quantize_seg3.py --ckpt <CKPT> --quant_mode calib --calib_batches 100 --calib_batch_size 8 --out_dir quantize_result

# 3) Quantized test + export .xmodel (writes to quantize_result/)
python eval_and_quantize_seg3.py --ckpt <CKPT> --quant_mode test --deploy --out_dir quantize_result

# 4) Compile for ZCU104 (DPUCZDX8G)
vai_c_xir \
  -x quantize_result/<MODEL>_int.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  -o /workspace/build \
  -n <MODEL>_P<PATCH>_zcu104