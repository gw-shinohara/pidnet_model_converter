# pidnet_to_onnx.py

import argparse
import os
import sys

# プロジェクトのルートパスを追加 (元の設定を維持)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.onnx import export
import onnx
import onnxruntime
import numpy as np

import models
from configs import config
from configs import update_config

def parse_args():
    parser = argparse.ArgumentParser(description='PIDNet to ONNX')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--model-file',
                        help='Path to the trained PyTorch model file',
                        required=True,
                        type=str)
    parser.add_argument('--output-file',
                        help='Path to save the ONNX model',
                        default='pidnet.onnx',
                        type=str)
    # 新しい引数を追加: デバッグモード
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable detailed ONNX mismatch analysis using strict tolerance (1e-5).')

    args = parser.parse_args()
    args.opts = []
    update_config(config, args)

    return args

def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    # --- 1. PyTorchモデルのロード ---
    print("=> PyTorchモデルをロード中...")
    num_classes = 19
    model = models.pidnet.get_pred_model(config, num_classes)

    # 学習済み重みをロード
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
        print(f"   - 重みファイル: {model_state_file}")
        pretrained_dict = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.eval()
    print("=> モデルのロード完了。")

    # 入力サイズは設定ファイルから取得
    height, width = config.TEST.IMAGE_SIZE
    dummy_input = torch.randn(1, 3, height, width)

    # --- 2. ONNXへのエクスポート ---
    output_file_path = args.output_file
    print(f"\n=> ONNXモデルをエクスポート中 ({output_file_path})...")

    try:
        # opuset_version 12 を使用
        export(model,
               dummy_input,
               output_file_path,
               export_params=True,
               opset_version=12,
               do_constant_folding=True,
               input_names=['input'],
               output_names=['output'],
               dynamic_axes={'input': {0: 'batch_size'},
                             'output': {0: 'batch_size'}})
        print("=> エクスポート完了。")
    except Exception as e:
        print(f"   - エラー: ONNXへのエクスポートに失敗しました。: {e}")
        return

    # --- 3. ONNXモデルの検証 (構造チェック) ---
    print("\n=> ONNXモデルを検証中...")
    try:
        onnx_model = onnx.load(output_file_path)
        onnx.checker.check_model(onnx_model)
        print("   - ONNXモデルの構造は正常です。")
    except Exception as e:
        print(f"   - エラー: ONNXモデルの検証に失敗しました。: {e}")
        return

    # --- 4. ONNX Runtimeでの最終推論と比較 (元の許容誤差) ---
    print("\n=> ONNX Runtimeでの最終推論テストを実行中...")
    try:
        ort_session = onnxruntime.InferenceSession(output_file_path)

        with torch.no_grad():
            torch_out = model(dummy_input)
            if isinstance(torch_out, tuple):
                torch_out = torch_out[0]

        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        # 元の許容誤差でテスト
        rtol_orig, atol_orig = 1e-02, 1e-04
        print(f"   - PyTorchとONNX Runtimeの最終出力を比較中 (rtol={rtol_orig}, atol={atol_orig})")
        np.testing.assert_allclose(torch_out.numpy(), ort_outs[0], rtol=rtol_orig, atol=atol_orig)
        print("   - PyTorchとONNX Runtimeの推論結果が許容誤差内で一致しました。✅")

    except Exception as e:
        # --- 5. デバッグモードでの中間ノード分析 ---
        print(f"   - エラー: 最終出力の比較に失敗しました。: {e}")
        if args.debug==True:
            print("\n=> デバッグモード: ONNXとPyTorchの中間ノードを詳細に分析中...")
            print("ONNXDebuggerをインポート")
            try:
                from utils.onnx_debugger import ONNXDebugger
            except ImportError:
                print("ERROR: onnx_debugger.py が見つかりません。")
                print("         onnx_debugger.py と pidnet_to_onnx.py を同じディレクトリに配置してください。")
            # ONNXDebuggerを初期化
            debugger = ONNXDebugger(model, output_file_path, dummy_input)

            # 厳密な誤差で分析を開始 (rtol=1e-05, atol=1e-08)
            debugger.analyze(rtol=1e-05, atol=1e-08)

            # ONNXノード名のリストも表示
            debugger.print_onnx_node_names()

        else:
            print("\n💡 ヒント: 最終比較でエラーが発生しています。詳細な分析を行うには、'--debug' フラグを付けて再実行してください。")


if __name__ == '__main__':
    main()
