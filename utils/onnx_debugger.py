# onnx_debugger.py
import torch
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np
from typing import Dict, Any

class ONNXDebugger:
    """
    PyTorchモデルとONNXモデルの中間出力を比較し、精度劣化の原因ノードを特定するためのツール。
    """
    def __init__(self, model: nn.Module, onnx_model_path: str, dummy_input: torch.Tensor):
        self.model = model
        self.onnx_model_path = onnx_model_path
        self.dummy_input = dummy_input
        self.pt_outputs: Dict[str, np.ndarray] = {}
        self.ort_outputs: Dict[str, np.ndarray] = {}

    def _register_and_run_pytorch(self):
        """
        PyTorchモデルにフックを登録し、推論を実行して中間出力をキャプチャします。
        """
        print("   - [PT] PyTorchモデルにフックを登録中...")
        hooks = []

        # Conv, BatchNorm, Upsampleなど、主要な演算子に絞ってフックを登録
        for name, module in self.model.named_modules():
            # PIDNetのような複雑なモデルでは、中間出力を取得するレイヤーを絞り込む
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.Upsample)):
                def get_hook(name):
                    def hook_fn(module, input, output):
                        # タプル出力の場合は最初の要素、またはテンソルをnumpyに変換
                        if isinstance(output, tuple):
                            self.pt_outputs[name] = output[0].detach().cpu().numpy()
                        elif isinstance(output, torch.Tensor):
                            self.pt_outputs[name] = output.detach().cpu().numpy()
                    return hook_fn

                hooks.append(module.register_forward_hook(get_hook(name)))

        # PyTorchで推論を実行し、中間出力を取得
        with torch.no_grad():
            self.model(self.dummy_input)

        # フックを削除
        for h in hooks:
            h.remove()
        print(f"   - [PT] PyTorchの中間出力を {len(self.pt_outputs)} 箇所でキャプチャしました。")


    def _run_onnx_runtime_all_outputs(self):
        """
        ONNX Runtimeで全中間ノードの出力を要求し、推論を実行します。
        """
        print("   - [ORT] ONNX Runtimeで全中間ノードの出力を要求中...")
        try:
            onnx_model = onnx.load(self.onnx_model_path)
        except Exception as e:
            print(f"   - [ORT] エラー: ONNXモデルのロードに失敗しました。: {e}")
            return

        # すべてのノードの出力名（中間ノード名）を取得
        all_output_names = [node_output for node in onnx_model.graph.node for node_output in node.output]

        # ONNX Runtimeセッションの作成
        try:
            ort_session = onnxruntime.InferenceSession(self.onnx_model_path)
        except Exception as e:
            print(f"   - [ORT] エラー: ONNX Runtimeセッションの作成に失敗しました。: {e}")
            return

        # ONNX Runtimeで推論を実行
        input_name = ort_session.get_inputs()[0].name
        ort_inputs = {input_name: self.dummy_input.numpy()}

        # 全出力ノードを指定して実行
        try:
            ort_outs = ort_session.run(all_output_names, ort_inputs)
            self.ort_outputs = dict(zip(all_output_names, ort_outs))
            print(f"   - [ORT] ONNX Runtimeの中間出力を {len(self.ort_outputs)} ノードでキャプチャしました。")
        except Exception as e:
            # ONNXグラフが大きすぎるとORTで全ノード出力を取得できない場合があります
            print(f"   - [ORT] エラー: ONNX Runtimeでの全ノード出力の取得に失敗しました。グラフが複雑すぎる可能性があります。: {e}")
            self.ort_outputs = {}

    def analyze(self, rtol: float = 1e-05, atol: float = 1e-08):
        """
        中間出力を比較し、最初の大きな不一致が発生したノードを特定します。
        """
        print("\n=== ONNX 精度劣化分析を開始 (デバッグモード) ===")

        self._register_and_run_pytorch()
        self._run_onnx_runtime_all_outputs()

        if not self.pt_outputs or not self.ort_outputs:
            print("\n🚨 分析を完了できませんでした。中間出力のキャプチャに失敗しました。")
            return

        print(f"\n- 厳密な誤差許容範囲: rtol={rtol}, atol={atol}")

        # PyTorchのモジュール出力とONNXノード出力を形状でマッチングして比較

        for pt_name, pt_out in self.pt_outputs.items():
            pt_shape = pt_out.shape

            # 対応するONNXノードを探す（形状が完全に一致するもの）
            matching_onnx_names = [name for name, ort_out in self.ort_outputs.items()
                                   if ort_out.shape == pt_shape]

            for onnx_name in matching_onnx_names:
                ort_out = self.ort_outputs[onnx_name]

                try:
                    # 厳密な比較を実行
                    np.testing.assert_allclose(pt_out, ort_out, rtol=rtol, atol=atol)
                except AssertionError as e:
                    # 最初に不一致が見つかったノードを報告
                    print("\n🚨 ------------------------------------------------------ 🚨")
                    print("🚨 最初の不一致が発生した可能性のあるノードを特定しました。")
                    print("🚨 ------------------------------------------------------ 🚨")
                    print(f"   - [PT モジュール]: **{pt_name}** (PyTorchでのレイヤー名)")
                    print(f"   - [ONNX ノード名]: **{onnx_name}** (ONNXグラフ上のノード出力名)")
                    print(f"   - [出力形状]: {pt_shape}")
                    max_abs_diff_match = next((line for line in str(e).split('\n') if 'Max absolute difference' in line), 'N/A')
                    print(f"   - [最大絶対誤差]: {max_abs_diff_match}")
                    print(f"   - [詳細]:\n{str(e)[:500]}...")
                    print("\n💡 次のステップ: **Netron**でONNXファイルを開き、このノード名とその周辺の演算子（Resize, Gatherなど）を確認してください。")
                    return

        print("\n✅ ------------------------------------------------------ ✅")
        print(f"✅ 厳密な閾値 (rtol={rtol}, atol={atol}) でも中間ノードの大きな不一致は見つかりませんでした。")
        print("✅ 誤差は主に最終層のわずかな累積か、非主要ノードの浮動小数点差によるものです。")
        print("✅ ------------------------------------------------------ ✅")

    def print_onnx_node_names(self):
        """ ONNXグラフ内の全中間ノード名を出力し、手動デバッグを補助します。 """
        print("\n--- ONNXグラフノード出力名のリスト（手動デバッグ用）---")
        if self.onnx_model_path:
            try:
                onnx_model = onnx.load(self.onnx_model_path)
                for i, node in enumerate(onnx_model.graph.node):
                    print(f"[{i+1:03d}] Op: {node.op_type} | Outputs: {', '.join(node.output)}")
                print("-----------------------------------------------------")
                print("ヒント: NetronでONNXファイルを可視化し、ここでリストアップされたノード名と比較してください。")
            except Exception as e:
                print(f"エラー: ONNXノード名のリストアップに失敗しました。: {e}")
