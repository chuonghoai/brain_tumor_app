import numpy as np
import cv2
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
from PIL import Image

class BaseGradCAM:
    def __init__(self, model):
        self.model = model

    def generate_heatmap(self, input_tensor):
        raise NotImplementedError("Phải triển khai ở lớp con.")

    def overlay_heatmap(self, original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
        img_array = np.array(original_image.convert('RGB'))
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.addWeighted(img_array, 1 - alpha, heatmap_color, alpha, 0)
        return Image.fromarray(superimposed_img)


class TorchGradCAM(BaseGradCAM):
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        super().__init__(model)
        self.model.eval()
        self.target_layer = target_layer
        
        if self.target_layer is None:
            self.target_layer = self._find_last_conv_layer()
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def _find_last_conv_layer(self):
        last_conv_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv_layer = module
        return last_conv_layer

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward()
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2)) 
        
        heatmap = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
            
        heatmap = np.maximum(heatmap, 0)
        heatmap_max = np.max(heatmap)
        if heatmap_max != 0:
            heatmap /= heatmap_max
            
        return heatmap

class KerasGradCAM(BaseGradCAM):
    def __init__(self, model):
        super().__init__(model)
        self.target_layer = self._find_target_layer()

    def _find_target_layer(self):
        try:
            import tensorflow as tf
        except ImportError:
            return None
            
        # Tìm nested model (thường là pre-trained base model)
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.Model):
                return layer
                
        # Ưu tiên tìm layer tên là 'bottleneck_conv'
        for layer in reversed(self.model.layers):
            if layer.name == 'bottleneck_conv':
                return layer
                
        # Hoặc tìm Conv2D cuối cùng
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer
        return None

    def generate_heatmap(self, input_array: np.ndarray, target_class: int = None) -> np.ndarray:
        if self.target_layer is None:
            return None
            
        try:
            import tensorflow as tf
            
            if isinstance(self.target_layer, tf.keras.Model):
                inner_model = self.target_layer
                inner_target = None
                for layer in reversed(inner_model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        inner_target = layer
                        break
                if inner_target is None:
                    return None
                    
                grad_model_inner = tf.keras.models.Model(
                    [inner_model.inputs],
                    [inner_target.output, inner_model.output]
                )
                
                with tf.GradientTape() as tape:
                    inputs = tf.cast(input_array, tf.float32)
                    conv_outputs, inner_outputs = grad_model_inner(inputs)
                    
                    x = inner_outputs
                    start_idx = self.model.layers.index(self.target_layer) + 1
                    for layer in self.model.layers[start_idx:]:
                        x = layer(x)
                    predictions = x
                    
                    if target_class is None:
                        target_class = tf.argmax(predictions[0])
                        
                    if len(predictions[0]) == 1:
                        loss = predictions[:, 0]
                    else:
                        loss = predictions[:, target_class]
            else:
                grad_model = tf.keras.models.Model(
                    [self.model.inputs],
                    [self.target_layer.output, self.model.output]
                )

                with tf.GradientTape() as tape:
                    inputs = tf.cast(input_array, tf.float32)
                    conv_outputs, predictions = grad_model(inputs)
                    
                    if target_class is None:
                        target_class = tf.argmax(predictions[0])
                        
                    if len(predictions[0]) == 1:
                        loss = predictions[:, 0]
                    else:
                        loss = predictions[:, target_class]

            grads = tape.gradient(loss, conv_outputs)
            
            # Global Average Pooling on gradients (axis 0 is batch, 1 is H, 2 is W, 3 is Channels)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ tf.expand_dims(pooled_grads, axis=-1)
            heatmap = tf.squeeze(heatmap)
            
            heatmap = tf.maximum(heatmap, 0)
            max_val = tf.math.reduce_max(heatmap)
            if max_val != 0:
                heatmap = heatmap / max_val
            return heatmap.numpy()
        except Exception as e:
            import logging
            logger = logging.getLogger("KerasGradCAM")
            logger.error(f"Lỗi khi tính Keras Grad-CAM: {e}")
            return None
