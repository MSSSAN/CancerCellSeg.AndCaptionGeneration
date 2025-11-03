"""
Medical Image Analysis Web App - RGB + Segmentation Caption Model
Streamlit application that performs:
1. Cell segmentation using HoVerNet + U-Net
2. Caption generation using RGB + Segmentation (4-channel input)

Input: PNG/JPG pathology image
Output: Segmentation visualization + Medical caption
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pickle
import nltk
from torchvision import transforms
import timm
import io
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import sys
import torch.nn.functional as F
from scipy.ndimage import center_of_mass

# Add paths for segmentation 
# 
# 
# \merged_folders\1.Source_code_for_AImodel\3. 병리 이미지 분할 모델_Is_this_segmentation\utils')

sys.path.append(r'.\pathSegmentation')
sys.path.append(r'.\utils')



from pathSeg.ml.hovernet import HoVerNet, post_process_batch_hovernet
import segmentation_models_pytorch as smp
from torchvision.transforms import ToTensor

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Page config
st.set_page_config(
    page_title="Medical Image Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #555;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .seg-box {
        background-color: #f0fff0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .step-indicator {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
IMG_SIZE = 512  # Changed from 1024 to 512
EMBED_SIZE = 300
HIDDEN_SIZE = 512  # FIXED: Changed from 256 to 512 to match trained model
NUM_LAYERS = 4
NUM_HEADS = 15
IMAGE_FEATURE_DIM = 1280  # EfficientNetV2-S output dimension

# Segmentation class configuration
CLASS_NAMES = ['Stroma', 'Immune', 'Normal/Epithelial', 'Tumor']
COLORS = {
    0: [1, 0.8, 0, 0.7],      # Yellow - Stroma
    1: [0, 1, 0, 0.7],         # Green - Immune
    2: [0, 0, 1, 0.7],         # Blue - Normal
    3: [1, 0, 0, 0.7],         # Red - Tumor
}


# ============================================================================
# SEGMENTATION FUNCTIONS
# ============================================================================

def post_process_unet(outputs):
    """Post-process U-Net output"""
    predict = F.softmax(outputs, dim=1)
    predict = predict.cpu().detach().numpy() > 0.5
    
    # Image size is now 512
    predict_temp = np.zeros((512, 512))
    for c in range(4):
        predict_temp = np.where(predict[0, c] == 1, c, predict_temp)
    
    return predict_temp


def combine_predictions(cell_seg, tissue_class):
    """Combine cell and tissue segmentation - adjusted for 512x512"""
    cell_total = np.zeros((512, 512, 4))
    
    for cell_id in range(1, int(np.max(cell_seg)) + 1):
        cell_mask = (cell_seg == cell_id).astype(int)
        center = center_of_mass(cell_mask)
        
        if np.isnan(center[0]) or np.isnan(center[1]):
            continue
        
        y, x = int(center[0]), int(center[1])
        tissue_type = int(tissue_class[y, x])
        cell_total[:, :, tissue_type] += cell_mask * cell_id
    
    return cell_total


def compress_segmentation_to_single_channel(segmentation):
    """
    Compress 4-channel segmentation to single channel (H, W)
    Values: 0=background, 1=Stroma, 2=Immune, 3=Normal, 4=Tumor
    
    Args:
        segmentation: (512, 512, 4) array
    
    Returns:
        seg_1ch: (512, 512) array with values 0-4
    """
    seg_1ch = np.zeros((512, 512), dtype=np.uint8)
    
    # Assign each pixel to the channel with maximum value
    for i in range(4):
        mask = segmentation[:, :, i] > 0
        seg_1ch[mask] = i + 1  # 1=Stroma, 2=Immune, 3=Normal, 4=Tumor
    
    return seg_1ch


def create_colored_overlay(segmentation, colors):
    """Create colored visualization of segmentation"""
    colored = np.zeros((512, 512, 4))
    
    for channel in range(4):
        mask = segmentation[:, :, channel] > 0
        for i in range(4):
            colored[:, :, i] = np.where(mask, colors[channel][i], colored[:, :, i])
    
    return colored


def preprocess_for_segmentation(image):
    """Preprocess image for segmentation"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    if image.size != (512, 512):
        image = image.resize((512, 512), Image.BILINEAR)
    
    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def run_segmentation(image, model1, model2, device):
    """Run segmentation inference on uploaded image"""
    # Preprocess
    image_tensor = preprocess_for_segmentation(image).float().to(device)
    
    # Run models
    with torch.no_grad():
        # Cell segmentation (HoVerNet)
        outputs1 = model1(image_tensor)
        cell_seg = post_process_batch_hovernet(outputs1, n_classes=None)[0]
        
        # Tissue classification (U-Net)
        outputs2 = model2(image_tensor)
        tissue_class = post_process_unet(outputs2)
    
    # Combine predictions
    final_segmentation = combine_predictions(cell_seg, tissue_class)
    
    return final_segmentation, cell_seg, tissue_class


def create_overlay_image(original_image, segmentation, show_mask=True, alpha=0.5):
    """
    Create overlay image as numpy array (for st.image display)
    
    Args:
        original_image: PIL Image
        segmentation: numpy array (512, 512, 4)
        show_mask: bool, whether to show the segmentation mask
        alpha: float, transparency of the mask overlay (0-1)
    
    Returns:
        tuple: (original_array, overlay_array)
    """
    # Convert PIL to numpy
    img_array = np.array(original_image.resize((512, 512)))
    
    if not show_mask:
        return img_array, img_array.copy()
    
    # Create colored overlay
    colored_mask = create_colored_overlay(segmentation, COLORS)
    colored_mask_rgb = (colored_mask[:, :, :3] * 255).astype(np.uint8)
    
    # Create blended overlay
    overlay_array = img_array.copy().astype(np.float32)
    
    for i in range(3):  # RGB channels
        mask_channel = colored_mask[:, :, 3] > 0
        overlay_array[:, :, i] = np.where(
            mask_channel,
            img_array[:, :, i] * (1 - alpha) + colored_mask_rgb[:, :, i] * alpha,
            img_array[:, :, i]
        )
    
    overlay_array = overlay_array.astype(np.uint8)
    
    return img_array, overlay_array


def calculate_segmentation_statistics(segmentation):
    """Calculate detailed segmentation statistics"""
    stats = {}
    
    for i, class_name in enumerate(CLASS_NAMES):
        channel = segmentation[:, :, i]
        num_regions = len(np.unique(channel)) - 1
        num_pixels = np.count_nonzero(channel)
        coverage = (num_pixels / (512 * 512)) * 100
        
        stats[class_name] = {
            'regions': num_regions,
            'pixels': num_pixels,
            'coverage': coverage
        }
    
    return stats


# ============================================================================
# CAPTION GENERATION MODELS - UPDATED FOR 4-CHANNEL INPUT
# ============================================================================

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx.get('<unk>', 0)
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


##############################################################################
# UPDATED FEATURE EXTRACTOR - NOW ACCEPTS 4 CHANNELS (RGB + SEGMENTATION)
##############################################################################
class FeatureExtractor(nn.Module):
    """
    Feature extractor using EfficientNetV2-S with 4-channel input
    Input: (B, 4, 512, 512) - RGB (3 channels) + Segmentation (1 channel)
    Output: (B, 1280) - Global average pooled features
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # Create EfficientNetV2-S with 4 input channels
        self.backbone = timm.create_model(
            'efficientnetv2_s',
            pretrained=False,  # Cannot use pretrained with 4 channels
            in_chans=4,        # CHANGED: 4 channels instead of 3
            num_classes=0,     # Remove classifier
            global_pool='avg'  # Global average pooling
        )

    def forward(self, x):
        # x: (B, 4, 512, 512) -> (B, 1280)
        return self.backbone(x)
##############################################################################


class AttentionMILModel(nn.Module):
    """Attention-based Multiple Instance Learning model - Updated for 4-channel input"""
    def __init__(self, num_classes, image_feature_dim, feature_extractor):
        super(AttentionMILModel, self).__init__()
        self.num_classes = num_classes
        self.image_feature_dim = image_feature_dim
        self.feature_extractor = feature_extractor
        
        self.attention = nn.Sequential(
            nn.Linear(image_feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classification_layer = nn.Linear(image_feature_dim, num_classes)

    def forward(self, inputs):
        # inputs: (B, 4, 512, 512)
        batch_size, channels, height, width = inputs.size()
        features = self.feature_extractor(inputs)  # (B, 1280)
        features = features.view(batch_size, -1)
        logits = self.classification_layer(features)
        return logits


class DecoderTransformer(nn.Module):
    """Transformer-based decoder for caption generation."""
    def __init__(self, embed_size, vocab_size, num_heads, hidden_size, num_layers, max_seq_length=100):
        super(DecoderTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, embed_size))
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_size
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def sample(self, features, max_seq_length=None):
        """Generate captions using greedy search with proper stopping."""
        if max_seq_length is None:
            max_seq_length = self.max_seq_length
        
        batch_size = features.size(0)
        sampled_ids = []
        
        # Start with <start> token (typically index 1)
        input_caption = torch.ones(batch_size, 1).long().to(features.device)
        memory = features.unsqueeze(0)
        
        # Track if we've hit end token
        has_ended = torch.zeros(batch_size, dtype=torch.bool).to(features.device)
        
        for i in range(max_seq_length):
            input_embedded = self.embed(input_caption) + \
                           self.positional_encoding[:, :input_caption.size(1), :]
            input_embedded = input_embedded.permute(1, 0, 2)
            tgt_mask = self.generate_square_subsequent_mask(input_embedded.size(0)).to(features.device)
            transformer_output = self.transformer_decoder(input_embedded, memory, tgt_mask=tgt_mask)
            transformer_output = transformer_output.permute(1, 0, 2)
            output = self.linear(transformer_output[:, -1, :])
            _, predicted = output.max(1)
            
            # Check for end token (typically index 2 or vocab specific)
            # End token is usually <end> with index 2 in most vocabularies
            is_end = (predicted == 2)  # <end> token
            has_ended = has_ended | is_end
            
            # If all sequences have ended, stop generating
            if has_ended.all():
                break
            
            sampled_ids.append(predicted)
            input_caption = torch.cat([input_caption, predicted.unsqueeze(1)], dim=1)
        
        sampled_ids = torch.stack(sampled_ids, 1)
        return sampled_ids


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def idx2word(vocab, indices):
    """Convert indices to words."""
    sentence = []
    aa = indices.cpu().numpy()
    for index in aa:
        word = vocab.idx2word[index]
        sentence.append(word)
    return sentence


def word2sentence(words_list):
    """Convert word list to sentence string."""
    sentence = ''
    for word in words_list:
        if word.isalnum():
            sentence += ' ' + word
        else:
            sentence += word
    return sentence.strip()


##############################################################################
# UPDATED PREPROCESSING - NOW CREATES 4-CHANNEL INPUT (RGB + SEG)
##############################################################################
def preprocess_for_caption(image, segmentation, device):
    """
    Preprocess image and segmentation for caption generation
    
    Args:
        image: PIL Image (original image)
        segmentation: numpy array (512, 512, 4) - 4-channel segmentation
        device: torch device
    
    Returns:
        tensor: (1, 4, 512, 512) - RGB (3 ch) + Segmentation (1 ch)
    """
    # Step 1: Preprocess RGB image
    transform_rgb = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    rgb_tensor = transform_rgb(image)  # (3, 512, 512)
    
    # Step 2: Compress segmentation to single channel
    seg_1ch = compress_segmentation_to_single_channel(segmentation)  # (512, 512)
    
    # Step 3: Convert segmentation to tensor and normalize [0, 1]
    seg_tensor = torch.from_numpy(seg_1ch.astype('float32')) / 4.0  # (512, 512)
    seg_tensor = seg_tensor.unsqueeze(0)  # (1, 512, 512)
    
    # Step 4: Concatenate RGB and Segmentation
    combined_tensor = torch.cat([rgb_tensor, seg_tensor], dim=0)  # (4, 512, 512)
    
    # Step 5: Handle NaN values
    combined_tensor = torch.nan_to_num(combined_tensor, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Step 6: Add batch dimension
    combined_tensor = combined_tensor.unsqueeze(0)  # (1, 4, 512, 512)
    
    return combined_tensor.to(device)
##############################################################################


def generate_caption(image, segmentation, vocab, feature_extractor, projection_layer, caption_decoder, device):
    """
    Generate caption for an image using RGB + Segmentation
    
    Args:
        image: PIL Image
        segmentation: numpy array (512, 512, 4)
        vocab: Vocabulary object
        feature_extractor: Feature extractor (4-channel)
        projection_layer: Linear layer (1280 -> 300)
        caption_decoder: Transformer decoder
        device: torch device
    
    Returns:
        caption_text: Generated caption string
        predicted_caption: List of words
    """
    # Preprocess image + segmentation to 4-channel input
    input_tensor = preprocess_for_caption(image, segmentation, device)  # (1, 4, 512, 512)
    
    # Generate caption
    with torch.no_grad():
        # Extract features
        features = feature_extractor(input_tensor)  # (1, 1280)
        
        # Project to embedding dimension
        features = projection_layer(features)  # (1, 300)
        
        # Generate caption
        sampled_ids = caption_decoder.sample(features)  # (1, max_seq_length)
    
    # Convert to words
    predicted_caption = idx2word(vocab, sampled_ids[0])
    
    # Remove special tokens and stop at end token
    clean_caption = []
    for word in predicted_caption:
        if word in ['<end>', '<pad>']:
            break  # Stop at end or pad token
        if word not in ['<start>', '<unk>']:
            clean_caption.append(word)
    
    # Convert to sentence
    caption_text = word2sentence(clean_caption)
    
    return caption_text, clean_caption


@st.cache_resource
def load_all_models(seg_model1_path, seg_model2_path, vocab_path, encoder_path, projection_path, decoder_path):
    """Load all models (segmentation + caption with projection layer)"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load segmentation models
    seg_model1 = HoVerNet(n_classes=None)
    seg_model1.load_state_dict(torch.load(seg_model1_path, map_location=device))
    seg_model1.to(device)
    seg_model1.eval()
    
    seg_model2 = smp.Unet(
        encoder_name="efficientnet-b7",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
    )
    seg_model2.load_state_dict(torch.load(seg_model2_path, map_location=device))
    seg_model2.to(device)
    seg_model2.eval()
    
    # Load vocabulary
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Load the combined checkpoint (contains both feature extractor and projection)
    checkpoint = torch.load(encoder_path, map_location=device)
    
    # Create feature extractor and load only its weights
    feature_extractor = FeatureExtractor().to(device)
    
    # Filter out only the backbone weights (remove 'proj.*' keys)
    feature_extractor_state = {k: v for k, v in checkpoint.items() if k.startswith('backbone.')}
    
    # Load with strict=False to ignore missing keys (conv_head, bn2 might be from different architecture)
    feature_extractor.load_state_dict(feature_extractor_state, strict=False)
    feature_extractor.eval()
    
    # Create projection layer
    projection_layer = nn.Linear(IMAGE_FEATURE_DIM, EMBED_SIZE).to(device)
    
    # Check if projection weights are in the encoder checkpoint or separate file
    if any(k.startswith('proj.') for k in checkpoint.keys()):
        # Projection is in the encoder checkpoint
        # The checkpoint has proj as a Sequential: proj.0, proj.1, proj.2, proj.3
        # We need to find which one is the actual Linear layer from 1280 to 300
        
        proj_keys = [k for k in checkpoint.keys() if k.startswith('proj.')]
        
        # Try to find the linear layer that matches our dimensions
        found_projection = False
        for i in range(10):  # Check proj.0 through proj.9
            weight_key = f'proj.{i}.weight'
            bias_key = f'proj.{i}.bias'
            
            if weight_key in checkpoint:
                weight = checkpoint[weight_key]
                # Check if this layer has the right shape (300, 1280)
                if weight.shape == torch.Size([EMBED_SIZE, IMAGE_FEATURE_DIM]):
                    projection_layer.weight.data = weight
                    projection_layer.bias.data = checkpoint[bias_key]
                    found_projection = True
                    break
        
        if not found_projection:
            # If not found in encoder, try loading from separate projection file
            if Path(projection_path).exists():
                projection_layer.load_state_dict(torch.load(projection_path, map_location=device))
    else:
        # Projection is in separate file
        if Path(projection_path).exists():
            projection_layer.load_state_dict(torch.load(projection_path, map_location=device))
    
    projection_layer.eval()
    
    # Decoder
    caption_decoder = DecoderTransformer(
        EMBED_SIZE, 
        len(vocab), 
        NUM_HEADS, 
        HIDDEN_SIZE, 
        NUM_LAYERS
    ).to(device)
    caption_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    caption_decoder.eval()
    
    return seg_model1, seg_model2, vocab, feature_extractor, projection_layer, caption_decoder, device


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    import pandas as pd
    
    # Initialize session state to store results
    if 'segmentation_result' not in st.session_state:
        st.session_state.segmentation_result = None
    if 'caption_result' not in st.session_state:
        st.session_state.caption_result = None
    if 'stats_result' not in st.session_state:
        st.session_state.stats_result = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Header
    st.markdown('<div class="main-header">🔬 Medical Image Analysis System</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cell Segmentation + AI Caption Generation (RGB + Seg Model)</div>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Segmentation Models")
        seg_model1_path = st.text_input(
            "HoVerNet (Cell Segmentation)",
            value="model/best_seg_BR_cell.pt"
        )
        seg_model2_path = st.text_input(
            "U-Net (Tissue Classification)",
            value="model/best_seg_BR_class.pt"
        )
        
        st.subheader("Caption Models (RGB + Seg)")
        vocab_path = st.text_input("Vocabulary File", value="vocab.pkl")
        encoder_path = st.text_input("Caption Encoder (4-ch)", value="encoder_4ch.pth")
        projection_path = st.text_input("Projection Layer", value="projection.pth")
        decoder_path = st.text_input("Caption Decoder", value="decoder.pth")
        
        st.markdown("---")
        
        st.subheader("Processing Options")
        show_detailed_vis = st.checkbox("Show Detailed Visualization", value=True)
        show_statistics = st.checkbox("Show Statistics", value=True)
        
        st.subheader("Visualization Options")
        show_mask_overlay = st.checkbox("Show Segmentation Mask", value=True, 
                                       help="Toggle segmentation overlay on/off")
        if show_mask_overlay:
            mask_alpha = st.slider("Mask Transparency", 
                                  min_value=0.0, 
                                  max_value=1.0, 
                                  value=0.5, 
                                  step=0.1,
                                  help="Adjust overlay transparency (0=invisible, 1=opaque)")
        
        st.markdown("---")
        
        st.subheader("ℹ️ About")
        st.info("""
        **New Model:** RGB + Segmentation
        
        **Segmentation:**
        - 🟡 Stroma (Connective tissue)
        - 🟢 Immune cells
        - 🔵 Normal/Epithelial cells
        - 🔴 Tumor cells
        
        **Caption Generation:**
        - 4-channel input (RGB + Seg)
        - EfficientNetV2-S encoder
        - Transformer decoder
        
        **Image size:** 512×512
        """)
        
        st.markdown("---")
        
        st.subheader("🖥️ System Info")
        device_type = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.write(f"**Device:** {device_type}")
        if torch.cuda.is_available():
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")
        st.write(f"**Image Size:** 512×512")
        st.write(f"**Input Channels:** 4 (RGB + Seg)")
    
    # Check if model files exist
    required_files = [
        (seg_model1_path, "HoVerNet model"),
        (seg_model2_path, "U-Net model"),
        (vocab_path, "Vocabulary"),
    ]
    
    missing_files = [(path, name) for path, name in required_files if not Path(path).exists()]
    
    if missing_files:
        st.warning("⚠️ Some model files are missing:")
        for path, name in missing_files:
            st.write(f"  - {name}: `{path}`")
        st.info("Please ensure all model files are in the correct location.")
        return
    
    # Load models
    with st.spinner("🔄 Loading models... This may take a moment."):
        try:
            seg_model1, seg_model2, vocab, feature_extractor, projection_layer, caption_decoder, device = load_all_models(
                seg_model1_path, seg_model2_path, vocab_path, encoder_path, projection_path, decoder_path
            )
            st.success("✅ All models loaded successfully! (4-channel caption model ready)")
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            st.exception(e)
            return
    
    # File upload
    st.markdown("---")
    st.header("📤 Upload Pathology Image")
    
    uploaded_file = st.file_uploader(
        "Choose a histopathology image",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a breast cancer pathology image for analysis"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display original image
        st.subheader("📷 Original Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        
        # Image info
        st.markdown("---")
        st.subheader("📊 Image Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Width", f"{image.size[0]} px")
        with col2:
            st.metric("Height", f"{image.size[1]} px")
        with col3:
            st.metric("Format", image.format or "Unknown")
        
        if image.size != (512, 512):
            st.info(f"ℹ️ Image will be resized from {image.size} to 512×512 for processing")
        
        # Process button
        st.markdown("---")
        
        if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Segmentation
                status_text.markdown('<div class="step-indicator">Step 1/2: Performing cell segmentation...</div>', 
                                   unsafe_allow_html=True)
                progress_bar.progress(10)
                
                with st.spinner("🔬 Running HoVerNet and U-Net..."):
                    segmentation, cell_seg, tissue_class = run_segmentation(
                        image, seg_model1, seg_model2, device
                    )
                    # Store in session state
                    st.session_state.segmentation_result = segmentation
                    st.session_state.uploaded_image = image
                
                progress_bar.progress(50)
                
                # Calculate statistics
                stats = calculate_segmentation_statistics(segmentation)
                st.session_state.stats_result = stats
                
                progress_bar.progress(60)
                
                # Step 2: Caption Generation with RGB + Segmentation
                status_text.markdown('<div class="step-indicator">Step 2/2: Generating caption (RGB + Seg)...</div>', 
                                   unsafe_allow_html=True)
                
                with st.spinner("📝 Generating AI caption with 4-channel input..."):
                    caption_text, caption_words = generate_caption(
                        image, segmentation, vocab, feature_extractor, projection_layer, caption_decoder, device
                    )
                    # Store in session state
                    st.session_state.caption_result = (caption_text, caption_words)
                
                progress_bar.progress(100)
                status_text.markdown('<div class="step-indicator">✅ Analysis Complete!</div>', 
                                   unsafe_allow_html=True)
                
                st.success("✅ Analysis complete! Use the controls in the sidebar to adjust visualization.")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
        
        # Display results if they exist (outside the button)
        if st.session_state.segmentation_result is not None:
            segmentation = st.session_state.segmentation_result
            stats = st.session_state.stats_result
            image = st.session_state.uploaded_image
            
            # Check if caption_result exists before unpacking
            if st.session_state.caption_result is not None:
                caption_text, caption_words = st.session_state.caption_result
            else:
                caption_text = "Caption generation pending..."
                caption_words = []
            
            # Display segmentation results
            st.markdown("---")
            st.markdown('<div class="seg-box"><h2>🧬 Segmentation Results</h2></div>', 
                      unsafe_allow_html=True)
            
            # Statistics
            if show_statistics:
                st.subheader("📈 Tissue Analysis")
                cols = st.columns(4)
                
                color_hex = ['#FFD700', '#00FF00', '#0000FF', '#FF0000']
                for i, (class_name, stat) in enumerate(stats.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3 style="margin:0; color: {color_hex[i]};">
                                {class_name}
                            </h3>
                            <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">
                                {stat['regions']}
                            </p>
                            <p style="margin: 0; color: #666;">regions</p>
                            <p style="margin: 0.5rem 0; color: #666;">
                                {stat['coverage']:.2f}% coverage
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Visualization (updates smoothly with toggle/slider changes)
            if show_detailed_vis:
                st.markdown("---")
                st.subheader("🎨 Segmentation Visualization")
                
                # Create overlay images
                alpha_value = mask_alpha if show_mask_overlay else 0
                original_array, overlay_array = create_overlay_image(
                    image, 
                    segmentation, 
                    show_mask=show_mask_overlay,
                    alpha=alpha_value
                )
                
                # Display using Streamlit columns (smoother than matplotlib)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(original_array, caption='Original Image', use_container_width=True)
                
                with col2:
                    if show_mask_overlay:
                        caption_text_img = f'Segmentation Overlay (α={alpha_value:.1f})'
                    else:
                        caption_text_img = 'Original Image (No Overlay)'
                    st.image(overlay_array, caption=caption_text_img, use_container_width=True)
                
                # Add legend using markdown
                st.markdown("""
                <div style="text-align: center; margin-top: 1rem;">
                    <span style="background-color: rgba(255, 204, 0, 0.7); padding: 5px 15px; margin: 5px; border-radius: 5px;">🟡 Stroma</span>
                    <span style="background-color: rgba(0, 255, 0, 0.7); padding: 5px 15px; margin: 5px; border-radius: 5px;">🟢 Immune</span>
                    <span style="background-color: rgba(0, 0, 255, 0.7); padding: 5px 15px; margin: 5px; border-radius: 5px; color: white;">🔵 Normal</span>
                    <span style="background-color: rgba(255, 0, 0, 0.7); padding: 5px 15px; margin: 5px; border-radius: 5px; color: white;">🔴 Tumor</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Display caption results below the images (only if caption was generated)
            if st.session_state.caption_result is not None:

                
                st.markdown(f"""
                <div class="result-box">
                    <p style="font-size: 1.4rem; line-height: 1.8; margin: 1rem 0; text-align: left; font-weight: 500;">
                        {caption_text}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("💡 This caption was generated using both RGB image and segmentation information (4-channel input)")
                
                # Caption details (collapsed by default)
                with st.expander("📋 View Caption Details"):
                    st.write("**Word-by-word breakdown:**")
                    st.write(", ".join(caption_words))
                    st.write(f"\n**Statistics:**")
                    st.write(f"- Total words: {len(caption_words)}")
                    st.write(f"- Character count: {len(caption_text)}")
                    st.write(f"- Input: 4 channels (RGB + Segmentation)")
                    st.write(f"- Model: EfficientNetV2-S + Transformer")
            
            # Download options
            st.markdown("---")
            st.subheader("💾 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download caption
                st.download_button(
                    label="📥 Caption (TXT)",
                    data=caption_text,
                    file_name="caption_rgb_seg.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Download segmentation NPY
                npy_buffer = io.BytesIO()
                np.save(npy_buffer, segmentation)
                npy_buffer.seek(0)
                
                st.download_button(
                    label="📥 Segmentation (NPY)",
                    data=npy_buffer,
                    file_name="segmentation.npy",
                    mime="application/octet-stream"
                )
            
            with col3:
                # Download full report
                report = f"""Medical Image Analysis Report (RGB + Seg Model)
=====================================

Image: {uploaded_file.name}
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Model: 4-channel (RGB + Segmentation)
Image Size: 512×512

SEGMENTATION RESULTS:
"""
                for class_name, stat in stats.items():
                    report += f"- {class_name}: {stat['regions']} regions ({stat['coverage']:.2f}% coverage)\n"
                
                report += f"""
AI GENERATED CAPTION (RGB + SEG):
{caption_text}

Caption Statistics:
- Word count: {len(caption_words)}
- Character count: {len(caption_text)}
- Input channels: 4 (RGB + Segmentation)
- Model: EfficientNetV2-S with 4-channel input
"""
                
                st.download_button(
                    label="📥 Full Report (TXT)",
                    data=report,
                    file_name="analysis_report_rgb_seg.txt",
                    mime="text/plain"
                )
    
    else:
        st.info("👆 Please upload a pathology image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Medical Image Analysis System</strong></p>
        <p>HoVerNet + U-Net Segmentation | RGB + Seg Caption Generation</p>
        <p>4-Channel Input Model (EfficientNetV2-S) | 512×512 Resolution</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
