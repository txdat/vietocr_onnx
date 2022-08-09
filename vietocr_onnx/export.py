import math
import os
import random
import traceback
import yaml
from functools import wraps
from typing import Optional, Any, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import onnx
import onnxsim
import onnxruntime as ort
from vietocr.tool.config import Cfg
from vietocr.tool.utils import download_weights
from vietocr.tool.translate import build_model

__all__ = ["export_vietocr_to_onnx"]

# fix seed
random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)
torch.cuda.manual_seed_all(42)

# onnx 's opset_version for torch's operators
# https://github.com/onnx/onnx/blob/main/docs/Operators.md
ONNX_OPSET_VERSION = 14


@torch.no_grad()
def try_export(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            print("=" * 80)
            print("=" + f" try exporting {f.__qualname__}")
            print("=" * 80)

            out = f(*args, **kwargs)

            print("=" * 80)
            print("=" + f" exported {f.__qualname__} done")
            print("=" * 80)
            print()

            return out

        except Exception as e:  # noqa
            print(traceback.format_exc())

            return False

    return wrapped


#####################################################
# export CNN Encoder
#####################################################


@try_export
def export_cnn(
    cnn: nn.Module,
    cfg: Dict[str, Any],
    export_dir: str,
    atol: float = 1e-5,
    prefix: str = "vietocr_",
) -> bool:
    """
    export vietocr 's cnn module to onnx

    fix onnx export issue with negative indices in transpose/permute functions
    # https://github.com/pbcquoc/vietocr/blob/79d284e4c5851af59b96c9f174b56470b9754ed3/vietocr/model/backbone/vgg.py#L40

    >>> #conv = conv.transpose(-1, -2)
    >>> #conv = conv.flatten(2)
    >>> #conv = conv.permute(-1, 0, 1)
    >>> conv = conv.permute(0, 1, 3, 2)
    >>> conv = conv.flatten(2)
    >>> conv = conv.permute(2, 0, 1)

    :param cnn:
    :param cfg:
    :param export_dir:
    :param atol:
    :param prefix:
    :return:
    """

    images_shape = (
        1,
        3,
        cfg["dataset"]["image_height"],  # stride 16
        cfg["dataset"]["image_max_width"],  # stride 4
    )
    images = torch.zeros(*images_shape).float()  # [B,C,H,maxW] -> [B,F,H//16,maxW//4]

    # export model
    export_path = f"{export_dir}/{prefix}cnn.onnx"
    torch.onnx.export(
        cnn,
        images,
        export_path,
        input_names=("images",),  # [B,C,H,W]
        output_names=("output",),  # [W//2,B,F]
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes={
            "images": {0: "batch", 3: "width"},
            "output": {0: "src_len", 1: "batch"},
        },
        verbose=False,
    )

    # check exported model
    model_onnx = onnx.load(export_path)  # noqa

    # remove initializers from inputs
    name_to_input = dict()
    for inp in model_onnx.graph.input:
        name_to_input[inp.name] = inp

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            model_onnx.graph.input.remove(name_to_input[initializer.name])

    onnx.checker.check_model(model_onnx)

    # simplify exported model
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={"images": images_shape},
    )
    onnx.save(model_onnx, export_path)

    assert check, "failed to simplify exported cnn module"

    # validate outputs
    images = np.random.uniform(0, 1, images_shape).astype(np.float32)

    torch_output = cnn(torch.tensor(images).float()).detach().cpu().numpy()

    sess = ort.InferenceSession(export_path)
    onnx_output = sess.run(
        ["output"], {"images": ort.OrtValue.ortvalue_from_numpy(images)}
    )[0]

    return np.allclose(torch_output, onnx_output, atol=atol)


#####################################################
# export Seq2Seq Encoder
#####################################################


@try_export
def export_seq2seq_encoder(
    encoder: nn.Module,
    cfg: Dict[str, Any],
    export_dir: str,
    atol: float = 1e-5,
    prefix: str = "vietocr_",
) -> bool:
    """
    export vietocr 's seq2seq 's encoder module to onnx

    :param encoder:
    :param cfg:
    :param export_dir:
    :param atol:
    :param prefix:
    :return:
    """

    src_shape = (cfg["dataset"]["image_max_width"] // 2, 1, cfg["cnn"]["hidden"])
    src = torch.zeros(*src_shape).float()  # [L,B,F], L = maxW//2

    # export model
    export_path = f"{export_dir}/{prefix}seq2seq_encoder.onnx"
    torch.onnx.export(
        encoder,
        src,
        export_path,
        input_names=("src",),
        output_names=("outputs", "hidden"),
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes={
            "src": {0: "src_len", 1: "batch"},  # [L,B,F]
            "outputs": {0: "src_len", 1: "batch"},  # [L,B,2*HEnc]
            "hidden": {0: "batch"},  # [B,HDec]
        },
        verbose=False,
    )

    # check exported model
    model_onnx = onnx.load(export_path)  # noqa

    # remove initializers from inputs
    name_to_input = dict()
    for inp in model_onnx.graph.input:
        name_to_input[inp.name] = inp

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            model_onnx.graph.input.remove(name_to_input[initializer.name])

    onnx.checker.check_model(model_onnx)

    # simplify exported model
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={"src": src_shape},
    )
    onnx.save(model_onnx, export_path)

    assert check, "failed to simplify exported seq2seq 's encoder module"

    # validate outputs
    src = np.random.normal(0, 1, src_shape).astype(np.float32)

    torch_outputs, torch_hidden = encoder(torch.tensor(src).float())
    torch_outputs = torch_outputs.detach().cpu().numpy()
    torch_hidden = torch_hidden.detach().cpu().numpy()

    sess = ort.InferenceSession(export_path)
    onnx_outputs, onnx_hidden = sess.run(
        ["outputs", "hidden"],
        {"src": ort.OrtValue.ortvalue_from_numpy(src)},
    )

    return np.allclose(torch_outputs, onnx_outputs, atol=atol) and np.allclose(
        torch_hidden, onnx_hidden, atol=atol
    )


@try_export
def compose_cnn_seq2seq_encoder(
    cfg: Dict[str, Any], export_dir: str, prefix: str = "vietocr_"
) -> bool:
    """
    compose exported vietocr 's seq2seq 's cnn and encoder 's onnx modules

    :param cfg:
    :param export_dir:
    :param prefix:
    :return:
    """

    # 1. load cnn and seq2seq 's encoder onnx models
    # cnn: "images" -> "output"
    cnn = onnx.load(f"{export_dir}/{prefix}cnn.onnx")
    # encoder: "src" -> {"outputs", "hidden"}
    encoder = onnx.load(f"{export_dir}/{prefix}seq2seq_encoder.onnx")

    # 2. compose 2 models
    # model: "cnn_images" -> {"seq2seq_encoder_outputs", "seq2seq_encoder_hidden"}
    model = onnx.compose.merge_models(
        cnn,
        encoder,
        io_map=[("output", "src")],
        prefix1="cnn_",  # params: "cnn_*"
        prefix2="seq2seq_encoder_",  # params: "seq2seq_encoder_*"
    )

    # 3. add beam tiling nodes to graph
    outputs_tiling_node = onnx.helper.make_node(
        "Tile",
        inputs=["seq2seq_encoder_outputs", "outputs_reps"],
        outputs=["tiled_outputs"],
    )
    hidden_tiling_node = onnx.helper.make_node(
        "Tile",
        inputs=["seq2seq_encoder_hidden", "hidden_reps"],
        outputs=["tiled_hidden"],
    )
    model.graph.node.append(outputs_tiling_node)
    model.graph.node.append(hidden_tiling_node)

    # 4. define new graph and model
    # inputs
    cnn_images = onnx.helper.make_tensor_value_info(
        "cnn_images", elem_type=onnx.TensorProto.FLOAT, shape=["batch", 3, 32, "width"]
    )
    # tile outputs [src_len,batch_size,2*HEnc] to [src_len,beam_size*batch_size,2*HEnc]
    outputs_reps = onnx.helper.make_tensor_value_info(
        "outputs_reps", elem_type=onnx.TensorProto.INT64, shape=[3]
    )
    # tile hidden [batch_size,HEnc] to [beam_size*batch_size,HEnc]
    hidden_reps = onnx.helper.make_tensor_value_info(
        "hidden_reps", elem_type=onnx.TensorProto.INT64, shape=[2]
    )

    # outputs
    tiled_outputs = onnx.helper.make_tensor_value_info(
        "tiled_outputs",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[
            "src_len",
            "batch",  # *beam_size
            2 * cfg["transformer"]["encoder_hidden"],
        ],  # combined with dim_param, dim_value
    )
    tiled_hidden = onnx.helper.make_tensor_value_info(
        "tiled_hidden",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[
            "batch",  # *beam_size
            cfg["transformer"]["decoder_hidden"],
        ],  # combined with dim_param, dim_value
    )

    # init new graph and model
    composed_graph = onnx.helper.make_graph(
        nodes=model.graph.node,
        name="vietocr_cnn_seq2seq_encoder",
        inputs=[cnn_images, outputs_reps, hidden_reps],
        outputs=[tiled_outputs, tiled_hidden],
        initializer=model.graph.initializer,
    )
    composed_model = onnx.helper.make_model(composed_graph)  # noqa
    composed_model.opset_import[0].version = ONNX_OPSET_VERSION

    # remove initializers from inputs
    name_to_input = dict()  # noqa
    for inp in composed_model.graph.input:
        name_to_input[inp.name] = inp

    for initializer in composed_model.graph.initializer:
        if initializer.name in name_to_input:
            composed_model.graph.input.remove(name_to_input[initializer.name])

    # 5. check model
    onnx.checker.check_model(composed_model)

    # 6. simplify model
    # model: {"cnn_images", "outputs_reps", "hidden_reps"} -> {"tiled_outputs", "tiled_hidden"}
    composed_model, check = onnxsim.simplify(
        composed_model,
        test_input_shapes={
            "cnn_images": (
                1,
                3,
                cfg["dataset"]["image_height"],  # stride 16
                cfg["dataset"]["image_max_width"],  # stride 4
            ),
            "outputs_reps": (3,),
            "hidden_reps": (2,),
        },
    )

    assert (
        check
    ), "failed to simplify composed module of seq2seq 's cnn and encoder modules"

    # 7. save model
    onnx.save(composed_model, f"{export_dir}/{prefix}cnn_seq2seq_encoder.onnx")

    return True


#####################################################
# export Seq2Seq Decoder
#####################################################


@try_export
def export_seq2seq_decoder(
    decoder: nn.Module,
    cfg: Dict[str, Any],
    export_dir: str,
    atol: float = 1e-5,
    prefix: str = "vietocr_",
) -> bool:
    """
    export vietocr 's seq2seq 's decoder module to onnx

    :param decoder:
    :param cfg:
    :param export_dir:
    :param atol:
    :param prefix:
    :return:
    """

    input_shape = (1,)
    hidden_shape = (1, cfg["transformer"]["decoder_hidden"])
    encoder_outputs_shape = (
        cfg["dataset"]["image_max_width"] // 2,
        1,
        2 * cfg["transformer"]["encoder_hidden"],
    )
    inp = torch.zeros(*input_shape).long()  # [B,]
    hidden = torch.zeros(*hidden_shape).float()  # [B,HDec]
    encoder_outputs = torch.zeros(
        *encoder_outputs_shape
    ).float()  # [L,B,2*HEnc], L = maxW//2

    # export model
    export_path = f"{export_dir}/{prefix}seq2seq_decoder.onnx"
    torch.onnx.export(
        decoder,
        (inp, hidden, encoder_outputs),
        export_path,
        input_names=("input", "hidden", "encoder_outputs"),
        output_names=(
            "prediction",
            "new_hidden",
            "attn",
        ),  # "new_hidden" is "hidden", "attn" is "a"
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes={
            "input": {0: "batch"},  # [B,]
            "hidden": {0: "batch"},  # [B,HDec]
            "encoder_outputs": {0: "src_len", 1: "batch"},  # [L,B,2*HEnc]
            "prediction": {0: "batch"},  # [B,vocab_size]
            "new_hidden": {0: "batch"},  # [B,HDec]
            "attn": {0: "batch"},  # [B,L]
        },
        verbose=False,
    )

    # check exported model
    model_onnx = onnx.load(export_path)  # noqa

    # remove initializers from inputs
    name_to_input = dict()
    for inp in model_onnx.graph.input:
        name_to_input[inp.name] = inp

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            model_onnx.graph.input.remove(name_to_input[initializer.name])

    onnx.checker.check_model(model_onnx)

    # simplify exported model
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={
            "input": input_shape,
            "hidden": hidden_shape,
            "encoder_outputs": encoder_outputs_shape,
        },
    )
    onnx.save(model_onnx, export_path)

    assert check, "failed to simplify exported seq2seq 's decoder module"

    # validate outputs
    inp = np.random.randint(0, len(cfg["vocab"]) + 4, input_shape).astype(np.longlong)
    hidden = np.random.normal(0, 1, hidden_shape).astype(np.float32)
    encoder_outputs = np.random.normal(0, 1, encoder_outputs_shape).astype(np.float32)

    torch_prediction, torch_new_hidden, torch_attn = decoder(
        torch.tensor(inp).long(),
        torch.tensor(hidden).float(),
        torch.tensor(encoder_outputs).float(),
    )
    torch_prediction = torch_prediction.detach().cpu().numpy()
    torch_new_hidden = torch_new_hidden.detach().cpu().numpy()
    torch_attn = torch_attn.detach().cpu().numpy()

    sess = ort.InferenceSession(export_path)
    onnx_prediction, onnx_new_hidden, onnx_attn = sess.run(
        ["prediction", "new_hidden", "attn"],
        {
            "input": ort.OrtValue.ortvalue_from_numpy(inp),
            "hidden": ort.OrtValue.ortvalue_from_numpy(hidden),
            "encoder_outputs": ort.OrtValue.ortvalue_from_numpy(encoder_outputs),
        },
    )

    return (
        np.allclose(torch_prediction, onnx_prediction, atol=atol)
        and np.allclose(torch_new_hidden, onnx_new_hidden, atol=atol)
        and np.allclose(torch_attn, onnx_attn, atol=atol)
    )


#####################################################
# export Transformer Encoder
#####################################################


@try_export
def export_transformer_pos_enc(
    pos_enc: nn.Module,
    cfg: Dict[str, Any],
    export_dir: str,
    atol: float = 1e-5,
    prefix: str = "vietocr_",
) -> bool:
    """
    export vietocr 's transformer 's positional encoding module to onnx

    modify `PositionalEncoding` at https://github.com/pbcquoc/vietocr/blob/79d284e4c5851af59b96c9f174b56470b9754ed3/vietocr/model/seqmodel/transformer.py#L79
    replacing for https://github.com/pbcquoc/vietocr/blob/79d284e4c5851af59b96c9f174b56470b9754ed3/vietocr/model/seqmodel/transformer.py#L39 and L42

    >>> class LanguageTransformer(nn.Module):
    >>>     def __init__(self, ...):  # noqa
    >>>         ...
    >>>
    >>>     def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):  # noqa
    >>>         ...
    >>>         # src = self.pos_enc(src*math.sqrt(self.d_model))
    >>>         # tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
    >>>         src = self.pos_enc(src)  # noqa
    >>>         tgt = self.pos_enc(self.embed_tgt(tgt))  # noqa
    >>> ...
    >>> class PositionalEncoding(nn.Module):
    >>>     def __init__(self, d_model, dropout=0.1, max_len=100):
    >>>         super(PositionalEncoding, self).__init__()
    >>>
    >>>         self.d_model_scale = math.sqrt(d_model)
    >>>         ...
    >>>
    >>>     def forward(self, x):
    >>>         x *= self.d_model_scale
    >>>         x = x + self.pe[:x.size(0), :]
    >>>         return self.dropout(x)

    :param pos_enc:
    :param cfg:
    :param export_dir:
    :param atol:
    :param prefix:
    :return:
    """

    src_shape = (cfg["dataset"]["image_max_width"] // 2, 1, cfg["cnn"]["hidden"])
    src = torch.zeros(*src_shape).float()  # [L,B,F], L = maxW//2

    # export model
    export_path = f"{export_dir}/{prefix}transformer_pos_enc.onnx"
    torch.onnx.export(
        pos_enc,
        src,
        export_path,
        input_names=("src",),
        output_names=("src_out",),
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes={
            "src": {0: "src_len", 1: "batch"},  # [L,B,F]
            "src_out": {0: "src_len", 1: "batch"},  # [L,B,F]
        },
        verbose=False,
    )

    # check exported model
    model_onnx = onnx.load(export_path)  # noqa

    # remove initializers from inputs
    name_to_input = dict()
    for inp in model_onnx.graph.input:
        name_to_input[inp.name] = inp

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            model_onnx.graph.input.remove(name_to_input[initializer.name])

    onnx.checker.check_model(model_onnx)

    # simplify exported model
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={"src": src_shape},
    )
    onnx.save(model_onnx, export_path)

    assert check, "failed to simplify exported transformer 's pos_enc module"

    # validate outputs
    src = np.random.normal(0, 1, src_shape).astype(np.float32)

    torch_src_out = pos_enc(torch.tensor(src).float())
    torch_src_out = torch_src_out.detach().cpu().numpy()

    sess = ort.InferenceSession(export_path)
    onnx_src_out = sess.run(
        ["src_out"],
        {"src": ort.OrtValue.ortvalue_from_numpy(src)},
    )[0]

    return np.allclose(torch_src_out, onnx_src_out, atol=atol)


@try_export
def export_transformer_encoder(
    encoder: nn.Module,
    cfg: Dict[str, Any],
    export_dir: str,
    atol: float = 1e-5,
    prefix: str = "vietocr_",
) -> bool:
    """
    export vietocr 's transformer 's encoder module to onnx

    :param encoder:
    :param cfg:
    :param export_dir:
    :param atol:
    :param prefix:
    :return:
    """

    src_shape = (cfg["dataset"]["image_max_width"] // 2, 1, cfg["cnn"]["hidden"])
    src = torch.zeros(*src_shape).float()  # [L,B,F], L = maxW//2

    # export model
    export_path = f"{export_dir}/{prefix}transformer_encoder.onnx"
    torch.onnx.export(
        encoder,
        src,
        export_path,
        input_names=("src",),
        output_names=("memory",),
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes={
            "src": {0: "src_len", 1: "batch"},  # [L,B,F]
            "memory": {0: "src_len", 1: "batch"},  # [L,B,HEnc]
        },
        verbose=False,
    )

    # check exported model
    model_onnx = onnx.load(export_path)  # noqa

    # remove initializers from inputs
    name_to_input = dict()
    for inp in model_onnx.graph.input:
        name_to_input[inp.name] = inp

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            model_onnx.graph.input.remove(name_to_input[initializer.name])

    onnx.checker.check_model(model_onnx)

    # simplify exported model
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={"src": src_shape},
    )
    onnx.save(model_onnx, export_path)

    assert check, "failed to simplify exported transformer 's encoder module"

    # validate outputs
    src = np.random.normal(0, 1, src_shape).astype(np.float32)

    torch_memory = encoder(torch.tensor(src).float())
    torch_memory = torch_memory.detach().cpu().numpy()

    sess = ort.InferenceSession(export_path)
    onnx_memory = sess.run(
        ["memory"],
        {"src": ort.OrtValue.ortvalue_from_numpy(src)},
    )[0]

    return np.allclose(torch_memory, onnx_memory, atol=atol)


@try_export
def compose_cnn_transformer_encoder(
    cfg: Dict[str, Any], export_dir: str, prefix: str = "vietocr_"
) -> bool:
    """
    compose exported vietocr 's transformer 's cnn and encoder 's onnx modules

    :param cfg:
    :param export_dir:
    :param prefix:
    :return:
    """

    # 1. load exported onnx models
    # cnn: "images" -> "output"
    cnn = onnx.load(f"{export_dir}/{prefix}cnn.onnx")
    # pos_enc: "src" -> "src_out"
    pos_enc = onnx.load(f"{export_dir}/{prefix}transformer_pos_enc.onnx")
    # encoder: "src" -> "memory"
    encoder = onnx.load(f"{export_dir}/{prefix}transformer_encoder.onnx")

    # 2. compose submodels
    model = onnx.compose.merge_models(
        cnn,
        pos_enc,
        io_map=[("output", "src")],
        prefix1="cnn_",  # params: "cnn_*"
        prefix2="transformer_pos_enc_",  # params: "transformer_pos_enc_*"
    )
    model = onnx.compose.merge_models(
        model,
        encoder,
        io_map=[("transformer_pos_enc_src_out", "src")],
        prefix2="transformer_encoder_",  # params: "transformer_encoder_*"
    )

    # 3. add beam tiling node
    memory_tiling_node = onnx.helper.make_node(
        "Tile",
        inputs=["transformer_encoder_memory", "memory_reps"],
        outputs=["tiled_memory"],
    )
    model.graph.node.append(memory_tiling_node)

    # 4. define new graph and model
    # inputs
    cnn_images = onnx.helper.make_tensor_value_info(
        "cnn_images", elem_type=onnx.TensorProto.FLOAT, shape=["batch", 3, 32, "width"]
    )
    # tile memory [src_len,batch_size,Henc] to [src_len,beam_size*batch_size,Henc]
    memory_reps = onnx.helper.make_tensor_value_info(
        "memory_reps", elem_type=onnx.TensorProto.INT64, shape=[3]
    )

    # outputs
    tiled_memory = onnx.helper.make_tensor_value_info(
        "tiled_memory",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[
            "src_len",
            "batch",  # *beam_size
            cfg["transformer"]["d_model"],
        ],  # combined with dim_param, dim_value
    )

    # init new graph and model
    composed_graph = onnx.helper.make_graph(
        nodes=model.graph.node,
        name="vietocr_cnn_transformer_encoder",
        inputs=[cnn_images, memory_reps],
        outputs=[tiled_memory],
        initializer=model.graph.initializer,
    )
    composed_model = onnx.helper.make_model(composed_graph)  # noqa
    composed_model.opset_import[0].version = ONNX_OPSET_VERSION

    # remove initializers from inputs
    name_to_input = dict()
    for inp in composed_model.graph.input:
        name_to_input[inp.name] = inp

    for initializer in composed_model.graph.initializer:
        if initializer.name in name_to_input:
            composed_model.graph.input.remove(name_to_input[initializer.name])

    # 5. check model
    onnx.checker.check_model(composed_model)

    # 6. simplify model
    # model: {"cnnimages", "memory_reps"} -> "tiled_memory"
    composed_model, check = onnxsim.simplify(
        composed_model,
        test_input_shapes={
            "cnn_images": (
                1,
                3,
                cfg["dataset"]["image_height"],  # stride 16
                cfg["dataset"]["image_max_width"],  # stride 4
            ),
            "memory_reps": (3,),
        },
    )

    assert (
        check
    ), "failed to simplify composed module of transformer 's cnn and encoder modules"

    # 7. save model
    onnx.save(composed_model, f"{export_dir}/{prefix}cnn_transformer_encoder.onnx")

    return True


#####################################################
# export Transformer Decoder
#####################################################


@try_export
def export_transformer_embed_tgt(
    embed_tgt: nn.Module,
    cfg: Dict[str, Any],
    export_dir: str,
    atol: float = 1e-5,
    prefix: str = "vietocr_",
) -> bool:
    """
    export vietocr 's transformer 's embeddings of targets module to onnx

    :param embed_tgt:
    :param cfg:
    :param export_dir:
    :param atol:
    :param prefix:
    :return:
    """

    # tgt is target's vocab index at each decoding step
    # TODO: check various decoding steps
    Ldec = 128
    tgt_shape = (Ldec, 1)
    tgt = torch.zeros(
        *tgt_shape
    ).long()  # [L,B], L is tgt 's length at each decoding step

    # export model
    export_path = f"{export_dir}/{prefix}transformer_embed_tgt.onnx"
    torch.onnx.export(
        embed_tgt,
        tgt,
        export_path,
        input_names=("tgt",),
        output_names=("tgt_out",),
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes={
            "tgt": {0: "tgt_len", 1: "batch"},  # [L,B]
            "tgt_out": {0: "tgt_len", 1: "batch"},  # [L,B,H]
        },
        verbose=False,
    )

    # check exported model
    model_onnx = onnx.load(export_path)  # noqa

    # remove initializers from inputs
    name_to_input = dict()
    for inp in model_onnx.graph.input:
        name_to_input[inp.name] = inp

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            model_onnx.graph.input.remove(name_to_input[initializer.name])

    onnx.checker.check_model(model_onnx)

    # simplify exported model
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={"tgt": tgt_shape},
    )
    onnx.save(model_onnx, export_path)

    assert check, "failed to simplify exported transformer 's embed_tgt module"

    # validate outputs
    tgt = np.random.randint(0, len(cfg["vocab"]) + 4, tgt_shape).astype(np.longlong)

    torch_tgt_out = embed_tgt(torch.tensor(tgt).long())
    torch_tgt_out = torch_tgt_out.detach().cpu().numpy()

    sess = ort.InferenceSession(export_path)
    onnx_tgt_out = sess.run(
        ["tgt_out"],
        {"tgt": ort.OrtValue.ortvalue_from_numpy(tgt)},
    )[0]

    return np.allclose(torch_tgt_out, onnx_tgt_out, atol=atol)


@try_export
def export_transformer_decoder(
    decoder: nn.Module,
    cfg: Dict[str, Any],
    export_dir: str,
    atol: float = 1e-5,
    prefix: str = "vietocr_",
) -> bool:
    """
    export vietocr 's transformer 's decoder module to onnx

    :param decoder:
    :param cfg:
    :param export_dir:
    :param atol:
    :param prefix:
    :return:
    """

    def gen_nopeek_mask(length: int) -> np.ndarray:
        # mask =
        # array([[0., -inf, -inf, -inf, -inf],
        #        [0.,   0., -inf, -inf, -inf],
        #        [0.,   0.,   0., -inf, -inf],
        #        [0.,   0.,   0.,   0., -inf],
        #        [0.,   0.,   0.,   0.,   0.]], dtype=float32)
        mask = np.zeros((length, length), dtype=np.float32)
        for i in range(length):
            mask[i, i + 1 :] = -math.inf

        return mask

    # TODO: check various decoding steps
    Ldec = 128
    tgt_shape = (
        Ldec,
        1,
        cfg["transformer"]["d_model"],
    )  # [L,B,Hemb], L is decoding step
    memory_shape = (
        cfg["dataset"]["image_max_width"] // 2,
        1,
        cfg["transformer"]["d_model"],
    )  # [Lsrc,B,Henc]
    tgt_mask_shape = (Ldec, Ldec)  # [L,L], L is decoding step
    tgt = torch.zeros(tgt_shape).float()
    memory = torch.zeros(memory_shape).float()
    tgt_mask = torch.tensor(gen_nopeek_mask(Ldec)).float()

    # export model
    export_path = f"{export_dir}/{prefix}transformer_decoder.onnx"
    torch.onnx.export(
        decoder,
        (tgt, memory, tgt_mask),
        export_path,
        input_names=("tgt", "memory", "tgt_mask"),
        output_names=("output",),
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes={
            "tgt": {0: "tgt_len", 1: "batch"},  # [L,B,Hemb]
            "memory": {0: "src_len", 1: "batch"},  # [Lsrc,B,Henc]
            "tgt_mask": {0: "tgt_len", 1: "tgt_len"},  # [L,L]
            "output": {0: "tgt_len", 1: "batch"},  # [L,B,Hdec]
        },
        verbose=False,
    )

    # check exported model
    model_onnx = onnx.load(export_path)  # noqa

    # remove initializers from inputs
    name_to_input = dict()
    for inp in model_onnx.graph.input:
        name_to_input[inp.name] = inp

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            model_onnx.graph.input.remove(name_to_input[initializer.name])

    onnx.checker.check_model(model_onnx)

    # simplify exported model
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={
            "tgt": tgt_shape,
            "memory": memory_shape,
            "tgt_mask": tgt_mask_shape,
        },
    )
    onnx.save(model_onnx, export_path)

    assert check, "failed to simplify exported transformer 's decoder module"

    # validate outputs
    tgt = np.random.normal(0, 1, tgt_shape).astype(np.float32)
    memory = np.random.normal(0, 1, memory_shape).astype(np.float32)
    tgt_mask = gen_nopeek_mask(Ldec)

    torch_output = decoder(
        torch.tensor(tgt).float(),
        torch.tensor(memory).float(),
        torch.tensor(tgt_mask).float(),
    )
    torch_output = torch_output.detach().cpu().numpy()

    sess = ort.InferenceSession(export_path)
    onnx_output = sess.run(
        ["output"],
        {
            "tgt": ort.OrtValue.ortvalue_from_numpy(tgt),
            "memory": ort.OrtValue.ortvalue_from_numpy(memory),
            "tgt_mask": ort.OrtValue.ortvalue_from_numpy(tgt_mask),
        },
    )[0]

    return np.allclose(torch_output, onnx_output, atol=atol)


@try_export
def export_transformer_fc(
    fc: nn.Module,
    cfg: Dict[str, Any],
    export_dir: str,
    atol: float = 1e-5,
    prefix: str = "vietocr_",
) -> bool:
    """
    export vietocr 's transformer 's fc module to onnx

    :param fc:
    :param cfg:
    :param export_dir:
    :param atol:
    :param prefix:
    :return:
    """

    # TODO: check various decoding steps
    Ldec = 128
    output_shape = (Ldec, 1, cfg["transformer"]["d_model"])  # [L,B,Hdec]
    output = torch.zeros(output_shape).float()

    # export model
    export_path = f"{export_dir}/{prefix}transformer_fc.onnx"
    torch.onnx.export(
        fc,
        output,
        export_path,
        input_names=("output",),
        output_names=("output_out",),
        opset_version=ONNX_OPSET_VERSION,
        dynamic_axes={
            "output": {0: "tgt_len", 1: "batch"},  # [L,B]
            "output_out": {0: "tgt_len", 1: "batch"},  # [L,B,vocab_size]
        },
        verbose=False,
    )

    # check exported model
    model_onnx = onnx.load(export_path)  # noqa

    # remove initializers from inputs
    name_to_input = dict()
    for inp in model_onnx.graph.input:
        name_to_input[inp.name] = inp

    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            model_onnx.graph.input.remove(name_to_input[initializer.name])

    onnx.checker.check_model(model_onnx)

    # simplify exported model
    model_onnx, check = onnxsim.simplify(
        model_onnx,
        test_input_shapes={"output": output_shape},
    )
    onnx.save(model_onnx, export_path)

    assert check, "failed to simplify exported transformer 's fc module"

    # validate outputs
    output = np.random.normal(0, 1, output_shape).astype(np.float32)

    torch_output_out = fc(torch.tensor(output).float())
    torch_output_out = torch_output_out.detach().cpu().numpy()

    sess = ort.InferenceSession(export_path)
    onnx_output_out = sess.run(
        ["output_out"],
        {"output": ort.OrtValue.ortvalue_from_numpy(output)},
    )[0]

    return np.allclose(torch_output_out, onnx_output_out, atol=atol)


@try_export
def compose_transformer_decoder(
    cfg: Dict[str, Any], export_dir: str, prefix: str = "vietocr_"
) -> bool:
    """
    compose exported vietocr 's transformer 's decoder 's onnx modules

    :param cfg:
    :param export_dir:
    :param prefix:
    :return:
    """

    # 1. load exported onnx models
    # embed_tgt: "tgt" -> "tgt_out"
    embed_tgt = onnx.load(f"{export_dir}/{prefix}transformer_embed_tgt.onnx")
    # pos_enc: "src" -> "src_out"
    pos_enc = onnx.load(f"{export_dir}/{prefix}transformer_pos_enc.onnx")
    # decoder: {"tgt", "memory", "tgt_mask"} -> "output"
    decoder = onnx.load(f"{export_dir}/{prefix}transformer_decoder.onnx")
    # fc: "output" -> "output_out"
    fc = onnx.load(f"{export_dir}/{prefix}transformer_fc.onnx")

    # 2. compose models
    model = onnx.compose.merge_models(
        embed_tgt,
        pos_enc,
        io_map=[("tgt_out", "src")],
        prefix1="transformer_embed_tgt_",  # params: "transformer_embed_tgt_*"
        prefix2="transformer_pos_enc_",  # params: "transformer_pos_enc_*"
    )
    model = onnx.compose.merge_models(
        model,
        decoder,
        io_map=[("transformer_pos_enc_src_out", "tgt")],
        prefix2="transformer_decoder_",  # params: "transformer_decoder_*"
    )
    model = onnx.compose.merge_models(
        model,
        fc,
        io_map=[("transformer_decoder_output", "output")],
        prefix2="transformer_fc_",  # params: "transformer_fc_*"
    )

    # 3. add slicing node after output to get last state only
    # create constant starts/ends nodes
    output_slicing_starts_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output_slicing_starts"],
        value=onnx.helper.make_tensor(
            name="output_slicing_starts_default",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[-1],  # get last state
        ),
    )
    output_slicing_ends_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["output_slicing_ends"],
        value=onnx.helper.make_tensor(
            name="output_slicing_ends_default",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[100000],  # INT_MAX?
        ),
    )
    output_slicing_node = onnx.helper.make_node(
        "Slice",
        inputs=[
            "transformer_fc_output_out",
            "output_slicing_starts",
            "output_slicing_ends",
        ],
        outputs=["sliced_output"],
    )
    model.graph.node.append(output_slicing_starts_node)
    model.graph.node.append(output_slicing_ends_node)
    model.graph.node.append(output_slicing_node)

    # 4. define new graph and model
    # output
    sliced_output = onnx.helper.make_tensor_value_info(
        "sliced_output",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[
            1,
            "batch",
            len(cfg["vocab"]) + 4,
        ],  # combined with dim_param, dim_value; 4 special tokens
    )

    # init new graph and model
    composed_graph = onnx.helper.make_graph(
        nodes=model.graph.node,
        name="vietocr_transformer_decoder",
        inputs=model.graph.input,
        outputs=[sliced_output],
        initializer=model.graph.initializer,
    )
    composed_model = onnx.helper.make_model(composed_graph)  # noqa
    composed_model.opset_import[0].version = ONNX_OPSET_VERSION

    # remove initializers from inputs
    name_to_input = dict()
    for inp in composed_model.graph.input:
        name_to_input[inp.name] = inp

    for initializer in composed_model.graph.initializer:
        if initializer.name in name_to_input:
            composed_model.graph.input.remove(name_to_input[initializer.name])

    # 5. check model
    onnx.checker.check_model(composed_model)

    # 6. simplify model
    # model: {"transformer_embed_tgt_tgt", "transformer_decoder_memory", "transformer_decoder_tgt_mask"}
    # -> "sliced_output"
    Ldec = 128
    composed_model, check = onnxsim.simplify(
        composed_model,
        test_input_shapes={
            "transformer_embed_tgt_tgt": (Ldec, 1),
            "transformer_decoder_memory": (
                cfg["dataset"]["image_max_width"] // 2,
                1,
                cfg["transformer"]["d_model"],
            ),
            "transformer_decoder_tgt_mask": (Ldec, Ldec),
        },
    )

    assert check, "failed to simplify composed module of transformer 's decoder modules"

    # 7. save model
    onnx.save(composed_model, f"{export_dir}/{prefix}transformer_decoder.onnx")

    return True


#####################################################
# export VietOCR to ONNX
#####################################################


def export_vietocr_to_onnx(
    export_dir: str,
    model_name: str,
    weights: Optional[str] = None,
    prefix: str = "vietocr",
):
    """
    export vietocr 's model to onnx
    :param export_dir:
    :param model_name: "vgg_seq2seq" or "vgg_transformer"
    :param weights: pretrained weights, use (downloaded) default weights if None
    :param prefix:
    :return:
    """

    os.makedirs(export_dir, exist_ok=True)

    assert model_name in {
        "vgg_seq2seq",
        "vgg_transformer",
    }, f"{model_name} is not supported ('vgg_seq2seq', 'vgg_transformer')"

    cfg = Cfg.load_config_from_name(model_name)

    # config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    # with open(f"{config_dir}/base.yaml", mode="r", encoding="utf-8") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # with open(f"{config_dir}/{model_name}.yaml", mode="r", encoding="utf-8") as f:
    #     cfg.update(yaml.load(f, Loader=yaml.FullLoader))

    cfg["cnn"]["pretrained"] = False  # download torchvision 's pretrained weights
    cfg["device"] = "cpu"

    if weights is None:
        weights = (
            download_weights(cfg["weights"])
            if cfg["weights"].startswith("http")
            else cfg["weights"]
        )

    model, _ = build_model(cfg)
    model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))
    model.eval()

    atol = 1e-5  # absolute tolerance

    assert export_cnn(
        model.cnn, cfg, export_dir, atol, prefix
    ), f"failed to export {model}'s cnn"

    if model_name == "vgg_seq2seq":
        assert export_seq2seq_encoder(
            model.transformer.encoder, cfg, export_dir, atol, prefix
        ), "failed to export seq2seq 's encoder"

        assert export_seq2seq_decoder(
            model.transformer.decoder, cfg, export_dir, atol, prefix
        ), "failed to export seq2seq 's decoder"

        assert compose_cnn_seq2seq_encoder(
            cfg, export_dir, prefix
        ), "failed to compose seq2seq 's cnn and encoder"

        for file in ["cnn", "seq2seq_encoder"]:
            os.remove(f"{export_dir}/{prefix}{file}.onnx")

    elif model_name == "vgg_transformer":
        assert export_transformer_pos_enc(
            model.transformer.pos_enc, cfg, export_dir, atol, prefix
        ), "failed to export transformer 's pos_enc"

        assert export_transformer_encoder(
            model.transformer.transformer.encoder, cfg, export_dir, atol, prefix
        ), "failed to export transformer 's encoder"

        assert export_transformer_embed_tgt(
            model.transformer.embed_tgt, cfg, export_dir, atol, prefix
        ), "failed to export transformer 's embed_tgt"

        # failed atol validation
        assert export_transformer_decoder(
            model.transformer.transformer.decoder, cfg, export_dir, 1e-4, prefix
        ), "failed to export transformer 's decoder"

        assert export_transformer_fc(
            model.transformer.fc, cfg, export_dir, atol, prefix
        ), "failed to export transformer 's fc"

        assert compose_cnn_transformer_encoder(
            cfg, export_dir, prefix
        ), "failed to compose transformer 's cnn and encoder"

        assert compose_transformer_decoder(
            cfg, export_dir, prefix
        ), "failed to compose transformer 's decoder"

        for file in [
            "cnn",
            "transformer_pos_enc",
            "transformer_encoder",
            "transformer_embed_tgt",
            "transformer_fc",
        ]:
            os.remove(f"{export_dir}/{prefix}{file}.onnx")

    print(f"exported vietocr 's {model_name} to {export_dir}!")


if __name__ == "__main__":
    export_vietocr_to_onnx(
        export_dir="/tmp/vietocr_seq2seq",
        model_name="vgg_seq2seq",
        weights="/home/txdat/Downloads/vgg_seq2seq.pth",
        prefix="",
    )

    export_vietocr_to_onnx(
        export_dir="/tmp/vietocr_transformer",
        model_name="vgg_transformer",
        weights="/home/txdat/Downloads/vgg_transformer.pth",
        prefix="",
    )
