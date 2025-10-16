
import logging
import base64
from io import BytesIO
from PIL import Image
import torch
# from torchvision.transforms.functional import to_pil_image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - optional dependency for newer transformers versions
    Qwen3VLForConditionalGeneration = None

func_to_enable_grad = '_sample'
setattr(LlavaForConditionalGeneration, func_to_enable_grad, torch.enable_grad(getattr(LlavaForConditionalGeneration, func_to_enable_grad)))

try:
    import intel_extension_for_pytorch as ipex
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

def get_processor_model(args):
    """Load processor + model pair with attention hooks for interpretability."""

    processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    if args.load_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif args.load_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quant_config = None

    model_kwargs = dict(
        quantization_config=quant_config,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
    )
    if quant_config is None:
        # Keep higher precision by default for interpretability gradients
        model_kwargs["torch_dtype"] = torch.bfloat16

    model_name_lower = args.model_name_or_path.lower()
    is_qwen = "qwen" in model_name_lower

    if is_qwen:
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError(
                "transformers.Qwen3VLForConditionalGeneration is unavailable. "
                "Please upgrade transformers to a version that includes Qwen3."
            )
        ModelClass = Qwen3VLForConditionalGeneration
    else:
        ModelClass = LlavaForConditionalGeneration

    model = ModelClass.from_pretrained(args.model_name_or_path, **model_kwargs)
    model._lvlim_model_type = "qwen" if is_qwen else "llava"

    if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "config"):
        if hasattr(model.vision_tower.config, "output_attentions"):
            model.vision_tower.config.output_attentions = True
    if hasattr(model, "visual") and hasattr(model.visual, "config"):
        if hasattr(model.visual.config, "output_attentions"):
            model.visual.config.output_attentions = True

    _register_attention_hooks(model)
    return processor, model


def _register_attention_hooks(model):
    """Attach forward hooks that capture attention tensors for later gradient processing."""

    model.enc_attn_weights = []
    model.enc_attn_weights_vit = []

    text_layers = _resolve_text_layers(model)
    vision_layers = _resolve_vision_layers(model)

    def _extract_attention_tensor(output):
        if isinstance(output, tuple):
            # (attn_output, attn_weights, ...)
            if len(output) >= 2:
                return output[1]
        elif isinstance(output, dict):
            for key in ("attn_weights", "attentions"):
                if key in output:
                    return output[key]
        return None

    def _make_hook(storage, warn_message):
        def _hook(module, inputs, output):  # noqa: D401 - signature defined by PyTorch
            attn_tensor = _extract_attention_tensor(output)
            if attn_tensor is None:
                if not getattr(module, "_lvlim_warned_missing_attn", False):
                    logger.warning(warn_message)
                    setattr(module, "_lvlim_warned_missing_attn", True)
                return output
            if isinstance(attn_tensor, (tuple, list)):
                attn_tensor = attn_tensor[0]
            if not isinstance(attn_tensor, torch.Tensor):
                if not getattr(module, "_lvlim_warned_attn_type", False):
                    logger.warning("Unexpected attention payload type %s", type(attn_tensor))
                    setattr(module, "_lvlim_warned_attn_type", True)
                return output
            attn_tensor.requires_grad_(True)
            attn_tensor.retain_grad()
            storage.append(attn_tensor)
            return output

        return _hook

    text_hook = _make_hook(
        model.enc_attn_weights,
        "Attention weights were not returned for the language model. "
        "Ensure output_attentions=True and disable flash attention backends for interpretability.",
    )
    vision_hook = _make_hook(
        model.enc_attn_weights_vit,
        "Attention weights were not returned for the vision model. "
        "Vision-level relevancy maps will be unavailable.",
    )

    model._lvlim_text_hooks = []
    for layer in text_layers:
        attn_module = getattr(layer, "self_attn", None) or getattr(layer, "self_attention", None)
        if attn_module is None:
            continue
        handle = attn_module.register_forward_hook(text_hook)
        model._lvlim_text_hooks.append(handle)

    model._lvlim_vision_hooks = []
    for layer in vision_layers:
        attn_module = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
        if attn_module is None:
            continue
        handle = attn_module.register_forward_hook(vision_hook)
        model._lvlim_vision_hooks.append(handle)


def _resolve_text_layers(model):
    """Return an iterable of decoder layers for the language backbone."""

    candidates = (
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("model", "language_model", "layers"),
    )

    for chain in candidates:
        obj = model
        for attr in chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj)
    logger.warning("Could not locate language model layers for attention hooks.")
    return []


def _resolve_vision_layers(model):
    """Return an iterable of vision encoder layers where attention can be captured."""

    candidates = (
        ("vision_tower", "vision_model", "encoder", "layers"),
        ("vision_tower", "vision_model", "layers"),
        ("visual", "blocks"),
        ("model", "visual", "blocks"),
    )

    for chain in candidates:
        obj = model
        for attr in chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return list(obj)
    logger.warning("Could not locate vision encoder layers for attention hooks.")
    return []

def process_image(image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
    if image_process_mode == "Pad":
        def expand2square(pil_img, background_color=(122, 116, 104)):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result
        image = expand2square(image)
    elif image_process_mode in ["Default", "Crop"]:
        pass
    elif image_process_mode == "Resize":
        image = image.resize((336, 336))
    else:
        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")
    if max(image.size) > max_len:
        max_hw, min_hw = max(image.size), min(image.size)
        aspect_ratio = max_hw / min_hw
        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
        longest_edge = int(shortest_edge * aspect_ratio)
        W, H = image.size
        if H > W:
            H, W = longest_edge, shortest_edge
        else:
            H, W = shortest_edge, longest_edge
        image = image.resize((W, H))
    if return_pil:
        return image
    else:
        buffered = BytesIO()
        image.save(buffered, format=image_format)
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        return img_b64_str


def to_gradio_chatbot(state):
        ret = []
        for i, (role, msg) in enumerate(state.messages):
            if i % 2 == 0:
                if type(msg) is tuple:
                    msg, image, image_process_mode = msg
                    img_b64_str = process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

def move_to_device(input, device='cpu'):

    if isinstance(input, torch.Tensor):
        return input.to(device).detach()
    elif isinstance(input, list):
        return [move_to_device(inp) for inp in input]
    elif isinstance(input, tuple):
        return tuple([move_to_device(inp) for inp in input])
    elif isinstance(input, dict):
        return dict( ((k, move_to_device(v)) for k,v in input.items()))
    else:
        raise ValueError(f"Unknown data type for {input.type}")
