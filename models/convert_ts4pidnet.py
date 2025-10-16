#!/usr/bin/env python3
import torch
import argparse
import os
from collections import OrderedDict
import sys
import yaml
import re

# Add the parent directory to the path to find the PIDNet module
# スクリプトの親ディレクトリをパスに追加してPIDNetモジュールを検索
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIDNet.pidnet import get_pred_model

def load_config_params(config_file_path):
    """
    設定ファイルからPIDNetに必要なパラメータ（モデルタイプとクラス数）を読み込みます。
    YAMLまたは単純なPython設定ファイルに対応。
    """
    if config_file_path.endswith('.yaml') or config_file_path.endswith('.yml'):
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)

            model_name = config.get('MODEL', {}).get('NAME', '').lower()
            num_classes = config.get('DATASET', {}).get('NUM_CLASSES', None)

    elif config_file_path.endswith('.py'):
        try:
            with open(config_file_path, 'r') as f:
                content = f.read()
            num_classes_match = re.search(r'NUM_CLASSES\s*=\s*(\d+)', content)
            model_name_match = re.search(r'model_name\s*=\s*[\'"]([^\'"]+)[\'"]', content)
            num_classes = int(num_classes_match.group(1)) if num_classes_match else None
            model_name = model_name_match.group(1).lower() if model_name_match else ''
        except Exception:
            raise ValueError("Failed to parse NUM_CLASSES or model_name from the Python config file.")
    else:
        raise ValueError("Unsupported configuration file format. Use .yaml, .yml, or .py.")

    if num_classes is None or not model_name:
        raise ValueError("Could not find required parameters (MODEL NAME or NUM_CLASSES) in the config file.")

    if 'pidnet_small' in model_name or 'pidnet-s' in model_name:
        model_type = 's'
    elif 'pidnet_medium' in model_name or 'pidnet-m' in model_name:
        model_type = 'm'
    elif 'pidnet_large' in model_name or 'pidnet-l' in model_name:
        model_type = 'l'
    else:
        raise NotImplementedError(f"Model type '{model_name}' is not a recognized PIDNet type (s, m, l).")

    print(f"INFO: Parameters extracted: PIDNet Type='{model_type}', Classes={num_classes}")
    return model_type, num_classes


def convert_pidnet_to_torchscript(
    checkpoint_path,
    model_type,
    num_classes,
    output_path,
    dummy_input_size=(1, 3, 1024, 1024)
):
    """
    PIDNetモデルのstate_dictをロードし、TorchScript形式で保存します。（pidnet.pyの定義に準拠した最終修正版）
    Loads a PIDNet model state_dict and saves it in TorchScript format. (Final corrected version based on pidnet.py)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: The input checkpoint file '{checkpoint_path}' does not exist.")
        return

    try:
        # 1. モデルのインスタンスを初期化
        model = get_pred_model(model_type, num_classes)

        # 2. チェックポイントを読み込み
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('state_dict', checkpoint)

        # 3. MMSegmentation形式のキーをネイティブのPIDNetモデルのキーにリマッピング（確定版ロジック）
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')

            # MMSegmentationの標準的な接頭辞を削除
            if name.startswith('backbone.'):
                name = name.replace('backbone.', '')
            if name.startswith('decode_head.'):
                name = name.replace('decode_head.', '')

            # --- pidnet.pyの定義に基づいた、具体的で順序付けされた置換ルール ---

            # Stem -> self.conv1
            # Note: MMSegmentationのstemは3チャンネル入力、ネイティブ実装は1チャンネル入力の場合があるため注意が必要だが、
            # Cityscapesの学習済みモデルは3チャンネルなので、`get_pred_model`で生成されるモデルも3チャンネル入力を想定
            name = name.replace('stem.0.conv.conv', 'conv1.0')
            name = name.replace('stem.0.bn.bn', 'conv1.1')
            name = name.replace('stem.1.conv.conv', 'conv1.3')
            name = name.replace('stem.1.bn.bn', 'conv1.4')

            # I-Branch -> self.layerX
            # MMSegmentationの`i_branch_layers`はネイティブモデルのlayer3,4,5に対応
            name = name.replace('i_branch_layers.0.', 'layer3.')
            name = name.replace('i_branch_layers.1.', 'layer4.')
            name = name.replace('i_branch_layers.2.', 'layer5.')

            # P-Branch -> self.layerX_
            name = name.replace('p_branch_layers.0.', 'layer3_.')
            name = name.replace('p_branch_layers.1.', 'layer4_.')
            name = name.replace('p_branch_layers.2.', 'layer5_.')

            # D-Branch -> self.layerX_d
            name = name.replace('d_branch_layers.0.', 'layer3_d.')
            name = name.replace('d_branch_layers.1.', 'layer4_d.')
            name = name.replace('d_branch_layers.2.', 'layer5_d.')

            # PAG Module -> self.pagX
            name = name.replace('pag_1.', 'pag3.')
            name = name.replace('pag_2.', 'pag4.')

            # DFM Module -> self.dfm
            name = name.replace('dfm.conv.conv', 'dfm.0')
            name = name.replace('dfm.bn.bn', 'dfm.1')

            # SPP Module -> self.spp
            if 'spp.scales' in name:
                for i in range(5):
                    name = name.replace(f'scales.{i}.conv.conv', f'scale{i}.0')
                    name = name.replace(f'scales.{i}.bn.bn', f'scale{i}.1')
                    name = name.replace(f'scales.{i}.1.conv.conv', f'scale{i}.1.0')
                    name = name.replace(f'scales.{i}.1.bn.bn', f'scale{i}.1.1')
            if 'spp.processes' in name:
                for i in range(4):
                    name = name.replace(f'processes.{i}.conv.conv', f'process{i+1}.0')
                    name = name.replace(f'processes.{i}.bn.bn', f'process{i+1}.1')
            name = name.replace('spp.compression.conv.conv', 'compression.0')
            name = name.replace('spp.compression.bn.bn', 'compression.1')
            name = name.replace('spp.shortcut.conv.conv', 'shortcut.0')
            name = name.replace('spp.shortcut.bn.bn', 'shortcut.1')

            # Prediction Head -> self.final_layer, self.seghead_p, self.seghead_d
            name = name.replace('conv_seg.', 'final_layer.conv2.')
            name = name.replace('i_head.', 'final_layer.') # i_headがfinal_layerの主要部分
            name = name.replace('p_head.', 'seghead_p.')

            # 汎用的なレイヤー名の正規化（特定の置換処理がすべて終わった後に実行）
            name = name.replace('.conv.conv', '.conv')
            name = name.replace('.bn.bn', '.bn')

            new_state_dict[name] = v

        # 4. 新しいstate_dictをモデルにロード
        print("INFO: Loading weights into model with strict=False...")
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

        # 5. モデルをトレースし、TorchScriptに変換
        print(f"INFO: Tracing model with input size {dummy_input_size}...")
        dummy_input = torch.randn(*dummy_input_size)
        traced_model = torch.jit.trace(model, dummy_input)

        # 6. TorchScriptモデルを保存
        traced_model.save(output_path)

        print(f"\nSUCCESS: Converted '{checkpoint_path}' to TorchScript and saved as '{output_path}'.")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during conversion: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a PIDNet PyTorch checkpoint to a TorchScript file using a config file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input .pth or .pt file.')
    parser.add_argument('--config-file', type=str, required=True, help='Path to the configuration file (.yaml or .py).')
    parser.add_argument('--output', type=str, required=True, help='Path to the output TorchScript .ts file.')
    parser.add_argument('--size', type=str, default='1,3,1024,1024', help='Comma-separated input tensor size (e.g., 1,3,512,512).')

    args = parser.parse_args()

    try:
        size_parts = [int(s.strip()) for s in args.size.split(',')]
        if len(size_parts) != 4:
            raise ValueError("Input size must have 4 components (N, C, H, W).")
        input_size = tuple(size_parts)
    except ValueError as e:
        print(f"Error: Invalid input size format. {e}")
        sys.exit(1)

    try:
        model_type, num_classes = load_config_params(args.config_file)
        convert_pidnet_to_torchscript(
            args.input,
            model_type,
            num_classes,
            args.output,
            dummy_input_size=input_size
        )
    except Exception as e:
        print(f"Error during execution: {e}")

