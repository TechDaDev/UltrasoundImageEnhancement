
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Subtract

# Define DnCNN model structure if needed (fallback)
def DnCNN():
    inpt = Input(shape=(None,None,1))
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   
    model = Model(inputs=inpt, outputs=x)
    return model

def denoise_image(model, img_path):
    # Load and preprocess
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape

    # Reflection padding (multiples of 4)
    pad_h = (4 - (h % 4)) % 4
    pad_w = (4 - (w % 4)) % 4
    arr_pad = np.pad(arr, ((0, pad_h), (0, pad_w)), mode='reflect')

    # Predict
    x = arr_pad.reshape(1, arr_pad.shape[0], arr_pad.shape[1], 1)
    y = model.predict(x, verbose=0)
    
    # Post-process
    out_full = y.reshape(arr_pad.shape)
    out = out_full[:h, :w]
    out = np.clip(out, 0.0, 1.0)
    
    return (out * 255.0).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description="DnCNN Denoising Script")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing noisy images")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, required=True, help="Path to DnCNN .h5 or .keras model")
    parser.add_argument("--save_mode", type=str, default="denoised", choices=["denoised", "comparison"], help="Save only denoised or comparison")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.model}...")
    try:
        model = load_model(args.model, compile=False)
    except Exception as e:
        print(f"Standard load failed: {e}. Trying to build structure and load weights...")
        model = DnCNN()
        model.load_weights(args.model)
        
    images = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']])
    print(f"Found {len(images)} images in {input_dir}")
    
    for i, img_path in enumerate(images):
        try:
            denoised_uint8 = denoise_image(model, img_path)
            
            out_name = img_path.stem + "_denoised" + img_path.suffix
            out_path = output_dir / out_name
            
            Image.fromarray(denoised_uint8).save(out_path)
            
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(images)}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            
    print("Done!")

if __name__ == "__main__":
    main()
